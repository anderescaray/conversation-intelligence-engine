import math

import pytest
from pipeline_fakes import FakeOperator, FakeVerifier, as_result, intent_output, make_conversation

from convintel.contracts.schemas import IntentOutput, OperatorResult, SentimentOutput, SummaryOutput
from convintel.errors import OperatorError
from convintel.operators.intent.operator import UNKNOWN_LABEL
from convintel.pipeline.graph import build_graph, route_after_intent
from convintel.pipeline.state import PipelineState

TAU = 0.861
# The graph does not look inside these outputs, so unvalidated placeholders are enough.
SENTIMENT = SentimentOutput.model_construct()
SUMMARY = SummaryOutput.model_construct()


def _state(intent: OperatorResult[IntentOutput] | None = None) -> PipelineState:
    return {
        "redacted_conversation": make_conversation(),
        "sentiment": None,
        "intent": intent,
        "summary": None,
        "errors": [],
    }


@pytest.mark.parametrize(
    ("label", "confidence", "expected"),
    [
        ("card_arrival", TAU, "end"),  # exactly τ stays with the classifier
        ("card_arrival", math.nextafter(TAU, 0), "verify"),  # just below τ
        ("card_arrival", 0.99, "end"),
        (UNKNOWN_LABEL, 0.10, "end"),  # unknown is never verified
    ],
)
def test_route_after_intent(label: str, confidence: float, expected: str) -> None:
    state = _state(as_result("intent", intent_output(label, confidence)))
    assert route_after_intent(state, TAU) == expected


def test_failed_classifier_is_not_sent_to_the_verifier() -> None:
    assert route_after_intent(_state(None), TAU) == "end"


def _graph(
    *, confidence: float, verifier: FakeVerifier, sentiment_error: OperatorError | None = None
):
    return build_graph(
        sentiment=FakeOperator("sentiment", SENTIMENT, error=sentiment_error),
        summary=FakeOperator("summary", SUMMARY),
        intent=FakeOperator("intent", intent_output(confidence=confidence)),
        verify=verifier,
        threshold=TAU,
    )


async def test_confident_intent_skips_the_verifier() -> None:
    verifier = FakeVerifier()
    final = await _graph(confidence=0.99, verifier=verifier).ainvoke(_state())

    assert verifier.calls == 0
    intent = final["intent"]
    assert intent is not None
    assert intent.output.label == "card_arrival"
    assert final["sentiment"] is not None
    assert final["summary"] is not None
    assert final["errors"] == []


async def test_low_confidence_intent_is_replaced_by_the_verifier() -> None:
    verified = as_result(
        "intent",
        intent_output("card_delivery_estimate", 0.5).model_copy(update={"source": "llm_verifier"}),
    )
    verifier = FakeVerifier(verified)
    final = await _graph(confidence=0.5, verifier=verifier).ainvoke(_state())

    assert verifier.calls == 1
    intent = final["intent"]
    assert intent is not None
    assert intent.output.label == "card_delivery_estimate"


async def test_failed_verification_keeps_the_classifier_label() -> None:
    verifier = FakeVerifier(None, errors=["intent_verifier:LLMError"])
    final = await _graph(confidence=0.5, verifier=verifier).ainvoke(_state())

    intent = final["intent"]
    assert intent is not None
    assert intent.output.label == "card_arrival"
    assert final["errors"] == ["intent_verifier:LLMError"]


async def test_a_failed_operator_fails_open_without_stopping_the_others() -> None:
    error = OperatorError("sentiment", "model unavailable")
    final = await _graph(confidence=0.99, verifier=FakeVerifier(), sentiment_error=error).ainvoke(
        _state()
    )

    assert final["sentiment"] is None
    assert final["summary"] is not None
    assert final["intent"] is not None
    assert final["errors"] == ["sentiment:OperatorError"]
