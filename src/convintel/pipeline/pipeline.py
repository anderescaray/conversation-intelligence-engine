"""The conversation pipeline: redact, analyze in a graph, aggregate.

PII redaction runs here, outside the graph and always, whatever the incoming conversation
claims: the graph's node inputs are recorded on spans, and a caller's ``redacted`` flag is a
claim, not a proof. A redaction failure stops the analysis (fail closed); every other
operator fails open inside the graph.
"""

import asyncio
import time
from collections.abc import Iterable
from typing import Any, Protocol, cast

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel

import convintel
from convintel.contracts.operators import Operator
from convintel.contracts.schemas import (
    Conversation,
    ConversationInsights,
    IntentOutput,
    OperatorResult,
    PIIOutput,
    SentimentOutput,
    SummaryOutput,
)
from convintel.errors import ConvIntelError, PipelineError
from convintel.telemetry import traced
from convintel.telemetry.metrics import PIPELINE_LATENCY_SECONDS

from .graph import VerifyFn, build_graph
from .state import PipelineState

_tracer = trace.get_tracer(__name__)


class WarmableOperator[T: BaseModel](Operator[T], Protocol):
    """An operator that loads (or validates) what it needs in ``warmup()``, at startup.

    Constructors stay cheap and do no I/O; a missing model or a bad LLM config must fail
    startup, not the first request.
    """

    async def warmup(self) -> None: ...


class IntentClassifier(WarmableOperator[IntentOutput], Protocol):
    """The intent operator plus what the pipeline needs at startup: its model's τ."""

    @property
    def threshold(self) -> float | None: ...


async def keep_classifier_label(
    conversation: Conversation, classified: OperatorResult[IntentOutput]
) -> tuple[OperatorResult[IntentOutput] | None, list[str]]:
    """Verifier placeholder until the LLM verifier exists: keeps the classifier's label."""
    return None, []


class Pipeline:
    def __init__(
        self,
        *,
        redactor: WarmableOperator[PIIOutput],
        sentiment: WarmableOperator[SentimentOutput],
        summary: WarmableOperator[SummaryOutput],
        intent: IntentClassifier,
        warmup_timeout_s: float,
        verify: VerifyFn = keep_classifier_label,
        threshold_override: float | None = None,
        version: str = convintel.__version__,
    ) -> None:
        """``warmup_timeout_s`` bounds all warmups together and has no default on purpose: how
        long startup may take is a deployment decision (``PIPELINE_WARMUP_TIMEOUT_S``), made
        by whoever composes the pipeline, never a value the pipeline invents.
        """
        self._redactor = redactor
        self._sentiment = sentiment
        self._summary = summary
        self._intent = intent
        self._warmup_timeout_s = warmup_timeout_s
        self._verify = verify
        self._threshold_override = threshold_override
        self._version = version
        self._graph: Any = None  # compiled in warmup(), once τ is known

    async def warmup(self) -> None:
        """Warm every operator, resolve τ and compile the graph. Raising aborts startup.

        The warmups run concurrently: they load independent models (or only check config),
        so startup takes the slowest one instead of their sum. They run in a TaskGroup: the
        first failure cancels the others, and every warmup that has already failed by then is
        reported together in one ExceptionGroup, so a broken deploy shows all its problems at
        once. Cancellation only stops waiting: work already handed to ``asyncio.to_thread``
        (model loading, the warmup inference) cannot be interrupted and runs to completion in
        its thread. On any failure the graph stays uncompiled, so ``analyze()`` keeps
        refusing to run. If the warmups together take longer than ``warmup_timeout_s``, they
        are cancelled and ``PipelineError("warmup_timeout")`` is raised; threads already
        running keep running, as above.
        """
        try:
            async with asyncio.timeout(self._warmup_timeout_s):
                async with asyncio.TaskGroup() as warmups:
                    warmups.create_task(self._redactor.warmup())
                    warmups.create_task(self._sentiment.warmup())
                    warmups.create_task(self._summary.warmup())
                    warmups.create_task(self._intent.warmup())
        except TimeoutError:
            # Only our deadline gets here bare: a warmup's own TimeoutError arrives wrapped in
            # the TaskGroup's ExceptionGroup.
            raise PipelineError("warmup_timeout") from None
        # `is not None`, not `or`: an override of 0.0 ("never verify") is a valid value.
        threshold = (
            self._threshold_override
            if self._threshold_override is not None
            else self._intent.threshold
        )
        if threshold is None:
            raise PipelineError("intent_threshold")
        self._graph = build_graph(
            sentiment=self._sentiment,
            summary=self._summary,
            intent=self._intent,
            verify=self._verify,
            threshold=threshold,
        )

    async def analyze(self, conversation: Conversation) -> ConversationInsights:
        if self._graph is None:
            raise RuntimeError("Pipeline.analyze() called before warmup()")
        start = time.perf_counter()
        try:
            # Not traced(): that records operator metrics. Exceptions keep their type only.
            with _tracer.start_as_current_span(
                "pipeline.analyze", record_exception=False, set_status_on_exception=False
            ) as root:
                root.set_attribute("conversation.id", conversation.id)
                root.set_attribute("pipeline.version", self._version)
                try:
                    return await self._analyze(conversation, root, start)
                except Exception as exc:
                    root.set_attribute("error.type", type(exc).__name__)
                    root.set_status(Status(StatusCode.ERROR))
                    raise
        finally:
            PIPELINE_LATENCY_SECONDS.labels(channel=conversation.channel.value).observe(
                time.perf_counter() - start
            )

    async def _analyze(
        self, conversation: Conversation, root: trace.Span, start: float
    ) -> ConversationInsights:
        pii = await self._redact(conversation)
        state: PipelineState = {
            "redacted_conversation": pii.output.redacted_conversation,
            "sentiment": None,
            "intent": None,
            "summary": None,
            "errors": [],
        }
        final = cast(PipelineState, await self._graph.ainvoke(state))
        total_cost, cost_errors = _total_cost(
            (pii, final["sentiment"], final["intent"], final["summary"])
        )
        span_context = root.get_span_context()
        return ConversationInsights(
            conversation_id=conversation.id,
            pipeline_version=self._version,
            sentiment=final["sentiment"],
            intent=final["intent"],
            summary=final["summary"],
            pii_entity_count=len(pii.output.entities),
            total_latency_ms=(time.perf_counter() - start) * 1000,
            total_cost_usd=total_cost,
            trace_id=format(span_context.trace_id, "032x") if span_context.is_valid else None,
            errors=[*final["errors"], *cost_errors],
        )

    async def _redact(self, conversation: Conversation) -> OperatorResult[PIIOutput]:
        """Always redact. Expected failures stop the pipeline; bugs propagate as they are."""
        try:
            with traced(f"operator.{self._redactor.name}", version=self._redactor.version):
                pii = await self._redactor.run(conversation)
        except ConvIntelError:
            # `from None`: the original error may quote the very text we failed to redact.
            raise PipelineError("pii") from None
        # The flag is checked on the redactor's output, where our own code sets it.
        if not pii.output.redacted_conversation.redacted:
            raise PipelineError("pii")
        return pii


def _total_cost(results: Iterable[OperatorResult[Any] | None]) -> tuple[float, list[str]]:
    """Sum the known costs. An LLM call (it used tokens) with no price is reported, so an
    unknown cost never passes for zero; local models have no tokens and no cost."""
    total = 0.0
    errors: list[str] = []
    for result in results:
        if result is None:
            continue
        if result.cost_usd is not None:
            total += result.cost_usd
        elif result.input_tokens is not None:
            errors.append(f"{result.operator}:UnknownCost")
    return total, errors
