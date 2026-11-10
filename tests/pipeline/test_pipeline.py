import asyncio
import re
import traceback
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pipeline_fakes import (
    REDACTED_TEXT,
    TAU,
    FakeIntentClassifier,
    FakeOperator,
    FakeRedactor,
    FakeVerifier,
    WarmupHook,
    intent_output,
    make_conversation,
)
from prometheus_client import REGISTRY
from pydantic import SecretStr

import convintel
from convintel.config import Settings
from convintel.contracts import Channel
from convintel.contracts.schemas import IntentOutput, SentimentOutput, SummaryOutput
from convintel.errors import OperatorError, PipelineError
from convintel.llm import LLMClient
from convintel.llm.testing import FakeStructuredChatModel, no_sleep, tool_answer
from convintel.pipeline.graph import IntentVerifier
from convintel.pipeline.pipeline import KeepClassifierLabel, Pipeline
from convintel.pipeline.verifier import LLMIntentVerifier, answer_model

SECRET = "marisol.q@example.com"
# The pipeline does not look inside these outputs, so unvalidated placeholders are enough.
SENTIMENT = SentimentOutput.model_construct()
SUMMARY = SummaryOutput.model_construct()


async def _never_finish() -> None:
    await asyncio.Event().wait()


class _Parts:
    def __init__(
        self,
        *,
        confidence: float = 0.95,
        bundle_threshold: float | None = TAU,
        redactor: FakeRedactor | None = None,
        sentiment_error: OperatorError | None = None,
        summary_cost: float | None = None,
        summary_tokens: int | None = None,
        failing_warmups: tuple[str, ...] = (),
        hanging_warmup: str | None = None,
        warmup_hook: WarmupHook | None = None,
        warmup_timeout_s: float = 300.0,
    ) -> None:
        self.warmup_timeout_s = warmup_timeout_s

        def warmup_error(name: str) -> OperatorError | None:
            return OperatorError(name, "model missing") if name in failing_warmups else None

        def hook(name: str) -> WarmupHook | None:
            return _never_finish if name == hanging_warmup else warmup_hook

        self.redactor = redactor or FakeRedactor(
            warmup_error=warmup_error("pii"), warmup_hook=hook("pii")
        )
        self.sentiment = FakeOperator(
            "sentiment",
            SENTIMENT,
            error=sentiment_error,
            warmup_error=warmup_error("sentiment"),
            warmup_hook=hook("sentiment"),
        )
        self.summary = FakeOperator(
            "summary",
            SUMMARY,
            cost_usd=summary_cost,
            input_tokens=summary_tokens,
            warmup_error=warmup_error("summary"),
            warmup_hook=hook("summary"),
        )
        self.intent = FakeIntentClassifier(
            intent_output(confidence=confidence),
            threshold=bundle_threshold,
            warmup_error=warmup_error("intent"),
            warmup_hook=hook("intent"),
        )
        self.verifier = FakeVerifier(
            warmup_error=warmup_error("intent_verifier"), warmup_hook=hook("intent_verifier")
        )

    def pipeline(self, *, threshold_override: float | None = None) -> Pipeline:
        return Pipeline(
            redactor=self.redactor,
            sentiment=self.sentiment,
            summary=self.summary,
            intent=self.intent,
            warmup_timeout_s=self.warmup_timeout_s,
            verify=self.verifier,
            threshold_override=threshold_override,
        )

    async def ready(self, *, threshold_override: float | None = None) -> Pipeline:
        pipeline = self.pipeline(threshold_override=threshold_override)
        await pipeline.warmup()
        return pipeline


# --- redaction: always, fail closed ---------------------------------------------------------


async def test_redacts_even_when_the_input_claims_to_be_redacted() -> None:
    parts = _Parts()
    insights = await (await parts.ready()).analyze(
        make_conversation(f"mail {SECRET}", redacted=True)
    )
    assert parts.redactor.calls == 1
    assert insights.pii_entity_count == 1


async def test_operators_only_ever_see_the_redacted_conversation() -> None:
    parts = _Parts()
    await (await parts.ready()).analyze(make_conversation(f"mail {SECRET}", redacted=False))
    for operator in (parts.sentiment, parts.summary, parts.intent):
        (seen,) = operator.seen
        assert seen.redacted
        assert SECRET not in repr(seen)


async def test_a_redaction_failure_stops_everything_without_its_message() -> None:
    parts = _Parts(redactor=FakeRedactor(error=OperatorError("pii", f"cannot parse {SECRET}")))
    pipeline = await parts.ready()

    with pytest.raises(PipelineError) as excinfo:
        await pipeline.analyze(make_conversation(f"mail {SECRET}"))

    error = excinfo.value
    assert error.stage == "pii"
    assert error.__cause__ is None
    assert error.__suppress_context__
    assert SECRET not in "".join(traceback.format_exception(error))
    assert parts.sentiment.calls == parts.summary.calls == parts.intent.calls == 0


async def test_a_redactor_that_forgets_the_flag_stops_everything() -> None:
    parts = _Parts(redactor=FakeRedactor(sets_flag=False))
    with pytest.raises(PipelineError):
        await (await parts.ready()).analyze(make_conversation())
    assert parts.sentiment.calls == 0


async def test_a_redactor_bug_propagates_as_a_bug() -> None:
    parts = _Parts(redactor=FakeRedactor(error=TypeError("bug")))
    with pytest.raises(TypeError):
        await (await parts.ready()).analyze(make_conversation())


# --- insights -----------------------------------------------------------------------------


async def test_insights_aggregate_the_branches() -> None:
    parts = _Parts(summary_cost=0.002, summary_tokens=120)
    insights = await (await parts.ready()).analyze(make_conversation(conversation_id="c-7"))

    assert insights.conversation_id == "c-7"
    assert insights.pipeline_version == convintel.__version__
    assert insights.sentiment is not None
    assert insights.summary is not None
    assert insights.intent is not None
    assert insights.total_cost_usd == pytest.approx(0.002)
    assert insights.total_latency_ms > 0
    assert insights.errors == []  # local models (no tokens) are not reported as unknown cost


async def test_an_unknown_llm_cost_is_reported_not_hidden() -> None:
    parts = _Parts(summary_cost=None, summary_tokens=120)
    insights = await (await parts.ready()).analyze(make_conversation())
    assert insights.total_cost_usd == 0
    assert insights.errors == ["summary:UnknownCost"]


async def test_errors_carry_only_the_operator_and_the_error_type() -> None:
    parts = _Parts(
        sentiment_error=OperatorError("sentiment", f"model choked on {SECRET}"),
        summary_tokens=10,
    )
    insights = await (await parts.ready()).analyze(make_conversation())

    assert insights.sentiment is None
    assert len(insights.errors) == 2
    assert all(re.fullmatch(r"[a-z_]+:[A-Za-z]+", e) for e in insights.errors)
    assert SECRET not in repr(insights)


def _latency_count(channel: str) -> float:
    sample = REGISTRY.get_sample_value(
        "convintel_pipeline_latency_seconds_count", {"channel": channel}
    )
    return sample or 0.0


async def test_latency_is_recorded_even_when_the_analysis_fails() -> None:
    before = _latency_count(Channel.CHAT.value)
    parts = _Parts(redactor=FakeRedactor(error=OperatorError("pii", "x")))
    with pytest.raises(PipelineError):
        await (await parts.ready()).analyze(make_conversation())
    assert _latency_count(Channel.CHAT.value) - before == 1


# --- warmup and τ -------------------------------------------------------------------------


async def test_the_bundle_threshold_is_used_without_an_override() -> None:
    parts = _Parts(confidence=0.7)  # below the bundle's τ
    await (await parts.ready()).analyze(make_conversation())
    assert parts.verifier.calls == 1


async def test_the_settings_threshold_overrides_the_bundle() -> None:
    parts = _Parts(confidence=0.7)
    await (await parts.ready(threshold_override=0.5)).analyze(make_conversation())
    assert parts.verifier.calls == 0


async def test_an_override_of_zero_means_never_verify() -> None:
    parts = _Parts(confidence=0.1)
    await (await parts.ready(threshold_override=0.0)).analyze(make_conversation())
    assert parts.verifier.calls == 0


def _root_attributes(spans: InMemorySpanExporter) -> dict[str, object]:
    (root,) = [s for s in spans.get_finished_spans() if s.name == "pipeline.analyze"]
    return dict(root.attributes or {})


@pytest.mark.parametrize(
    ("override", "expected"),
    [(None, TAU), (0.5, 0.5), (0.0, 0.0)],  # the bundle's τ, an override, "never verify"
)
async def test_the_effective_threshold_is_recorded_on_the_root_span(
    spans: InMemorySpanExporter, override: float | None, expected: float
) -> None:
    pipeline = await _Parts().ready(threshold_override=override)
    await pipeline.analyze(make_conversation(conversation_id="c-tau"))

    attributes = _root_attributes(spans)
    assert attributes["intent.threshold"] == expected
    assert attributes["conversation.id"] == "c-tau"


async def test_warmup_fails_without_any_threshold() -> None:
    with pytest.raises(PipelineError):
        await _Parts(bundle_threshold=None).ready()


async def test_analyze_before_warmup_is_a_bug() -> None:
    with pytest.raises(RuntimeError):
        await _Parts().pipeline().analyze(make_conversation())


async def test_warmup_warms_every_operator_and_the_verifier_once() -> None:
    parts = _Parts()
    await parts.ready()
    warmed = (parts.redactor, parts.sentiment, parts.summary, parts.intent, parts.verifier)
    assert [w.warmups for w in warmed] == [1, 1, 1, 1, 1]


def _operator_error(operator: str) -> pytest.RaisesExc[OperatorError]:
    return pytest.RaisesExc(OperatorError, check=lambda exc: exc.operator == operator)


@pytest.mark.parametrize("failing", ["pii", "sentiment", "summary", "intent", "intent_verifier"])
async def test_a_failing_warmup_aborts_startup_and_the_graph_is_not_compiled(
    failing: str,
) -> None:
    pipeline = _Parts(failing_warmups=(failing,)).pipeline()

    with pytest.RaisesGroup(_operator_error(failing)):
        await pipeline.warmup()

    with pytest.raises(RuntimeError, match="before warmup"):
        await pipeline.analyze(make_conversation())


async def test_every_failed_warmup_is_reported_together() -> None:
    pipeline = _Parts(failing_warmups=("pii", "summary")).pipeline()

    with pytest.RaisesGroup(_operator_error("pii"), _operator_error("summary")):
        await pipeline.warmup()

    with pytest.raises(RuntimeError, match="before warmup"):
        await pipeline.analyze(make_conversation())


async def test_warmups_run_concurrently() -> None:
    # Each warmup (four operators and the verifier) waits until all five have started: run one
    # after another, the first would wait forever, so only a concurrent warmup gets past.
    barrier = asyncio.Barrier(5)

    async def wait_for_the_others() -> None:
        await barrier.wait()

    pipeline = _Parts(warmup_hook=wait_for_the_others).pipeline()

    async with asyncio.timeout(1):
        await pipeline.warmup()


async def test_a_warmup_that_never_finishes_times_out() -> None:
    pipeline = _Parts(hanging_warmup="sentiment", warmup_timeout_s=0.05).pipeline()

    with pytest.raises(PipelineError, match="warmup_timeout") as excinfo:
        # Bounded on its own: if the deadline regresses, this fails instead of hanging.
        await asyncio.wait_for(pipeline.warmup(), timeout=2)

    assert excinfo.value.__suppress_context__
    with pytest.raises(RuntimeError, match="before warmup"):
        await pipeline.analyze(make_conversation())


# --- verifier -------------------------------------------------------------------------------


async def test_without_a_verifier_the_classifier_label_is_kept(
    spans: InMemorySpanExporter,
) -> None:
    parts = _Parts(confidence=0.7)  # below τ: routed to the verifier
    pipeline = Pipeline(
        redactor=parts.redactor,
        sentiment=parts.sentiment,
        summary=parts.summary,
        intent=parts.intent,
        warmup_timeout_s=parts.warmup_timeout_s,
    )
    await pipeline.warmup()

    insights = await pipeline.analyze(make_conversation())

    assert insights.intent is not None
    assert insights.intent.output.label == "card_arrival"
    assert insights.intent.output.source == "classifier"
    assert insights.errors == []
    (span,) = [s for s in spans.get_finished_spans() if s.name == "operator.intent_verifier"]
    assert dict(span.attributes or {})["operator.version"] == "placeholder"


def test_keep_classifier_label_is_an_intent_verifier() -> None:
    verifier: IntentVerifier = KeepClassifierLabel()  # pyright checks the protocol here
    assert verifier.version == "placeholder"


async def test_the_llm_verifier_result_reaches_the_insights() -> None:
    # The real verifier and client with a scripted model: no network, no paid call.
    top_k = [("card_arrival", 0.6), ("card_delivery_estimate", 0.3)]
    parts = _Parts()
    parts.intent = FakeIntentClassifier(
        IntentOutput(label="card_arrival", confidence=0.6, source="classifier", top_k=top_k)
    )
    model = FakeStructuredChatModel(
        responses=[tool_answer(answer_model(2), {"reasoning": "SENTINEL-WHY", "choice": "NONE"})]
    )
    llm: dict[str, Any] = {
        "llm_provider": "anthropic",
        "llm_model": "claude-haiku-4-5",
        "anthropic_api_key": SecretStr("test-key"),
    }
    settings = Settings(_env_file=None, **llm)  # pyright: ignore[reportCallIssue]
    pipeline = Pipeline(
        redactor=parts.redactor,
        sentiment=parts.sentiment,
        summary=parts.summary,
        intent=parts.intent,
        warmup_timeout_s=parts.warmup_timeout_s,
        verify=LLMIntentVerifier(LLMClient(settings, chat_model=model, sleep=no_sleep)),
    )
    await pipeline.warmup()

    insights = await pipeline.analyze(make_conversation(f"mail {SECRET}"))

    assert insights.intent is not None
    assert insights.intent.output.label == "unknown"
    assert insights.intent.output.source == "llm_verifier"
    assert insights.total_cost_usd == pytest.approx(insights.intent.cost_usd)
    assert insights.total_cost_usd > 0
    assert insights.errors == []
    # Only the redacted conversation reaches the LLM, and the reasoning never comes back.
    (messages,) = model.calls
    assert SECRET not in str(messages)
    assert REDACTED_TEXT in str(messages[-1].content)
    assert "SENTINEL-WHY" not in insights.model_dump_json()
