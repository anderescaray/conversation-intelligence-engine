"""The conversation pipeline graph.

Sentiment, summary and intent run in parallel on the redacted conversation. A
low-confidence intent goes to the verifier, a second opinion over the
classifier's top-k. The graph ends when every branch has ended, so there is no
aggregate node: ``Pipeline.analyze()`` builds the insights from the final state.
"""

from typing import Literal, Protocol

from langgraph.graph import END, START, StateGraph

from convintel.contracts.operators import Operator
from convintel.contracts.schemas import (
    Conversation,
    IntentOutput,
    OperatorResult,
    SentimentOutput,
    SummaryOutput,
)
from convintel.operators.intent.operator import UNKNOWN_LABEL
from convintel.telemetry import traced

from .nodes import run_operator
from .state import PipelineState, PipelineUpdate


class IntentVerifier(Protocol):
    """Second opinion on a low-confidence intent.

    A call returns the verified result (None if verification failed) and the errors to record;
    a failed verification keeps the classifier's result. The pipeline warms the verifier at
    startup with the operators, and the graph traces every call as ``operator.intent_verifier``,
    whichever implementation is plugged in.
    """

    @property
    def version(self) -> str: ...

    async def warmup(self) -> None: ...

    async def __call__(
        self, conversation: Conversation, classified: OperatorResult[IntentOutput]
    ) -> tuple[OperatorResult[IntentOutput] | None, list[str]]: ...


type Route = Literal["verify", "end"]


def route_after_intent(state: PipelineState, threshold: float) -> Route:
    """Verify only a real label whose confidence is strictly below τ.

    Strictly below: τ was calibrated with ``confidence >= τ`` meaning "served by
    the classifier" (``operators/intent/evaluation.py``), so the boundary stays on
    the classifier's side. A failed classifier (None) is not rescued by the
    verifier: it fails open and shows up in the error metrics, instead of turning
    an outage into LLM traffic.
    """
    result = state["intent"]
    if result is None or result.output.label == UNKNOWN_LABEL:
        return "end"
    return "verify" if result.output.confidence < threshold else "end"


def build_graph(
    *,
    sentiment: Operator[SentimentOutput],
    summary: Operator[SummaryOutput],
    intent: Operator[IntentOutput],
    verify: IntentVerifier,
    threshold: float,
):
    """Compile the graph. Operators, verifier and τ are injected, so tests use fakes
    and the τ policy (settings override or bundle) is resolved by the caller."""

    async def run_sentiment(state: PipelineState) -> PipelineUpdate:
        result, errors = await run_operator(sentiment, state["redacted_conversation"])
        return {"sentiment": result, "errors": errors}

    async def run_summary(state: PipelineState) -> PipelineUpdate:
        result, errors = await run_operator(summary, state["redacted_conversation"])
        return {"summary": result, "errors": errors}

    async def run_intent(state: PipelineState) -> PipelineUpdate:
        result, errors = await run_operator(intent, state["redacted_conversation"])
        return {"intent": result, "errors": errors}

    async def verify_intent(state: PipelineState) -> PipelineUpdate:
        classified = state["intent"]
        if classified is None:  # unreachable: the router only sends a classifier result
            raise RuntimeError("verifier reached without a classifier result")
        # Same span name and metrics as the operators; the verifier only annotates the span.
        with traced("operator.intent_verifier", version=verify.version):
            verified, errors = await verify(state["redacted_conversation"], classified)
        if verified is None:
            return {"errors": errors}  # verification failed: keep the classifier's label
        return {"intent": verified, "errors": errors}

    def route(state: PipelineState) -> Route:
        return route_after_intent(state, threshold)

    graph = StateGraph(PipelineState)
    graph.add_node("run_sentiment", run_sentiment)
    graph.add_node("run_summary", run_summary)
    graph.add_node("run_intent", run_intent)
    graph.add_node("verify_intent", verify_intent)
    graph.add_edge(START, "run_sentiment")
    graph.add_edge(START, "run_summary")
    graph.add_edge(START, "run_intent")
    graph.add_edge("run_sentiment", END)
    graph.add_edge("run_summary", END)
    graph.add_conditional_edges("run_intent", route, {"verify": "verify_intent", "end": END})
    graph.add_edge("verify_intent", END)
    return graph.compile()
