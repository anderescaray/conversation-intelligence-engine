"""Shared test doubles for the pipeline tests."""

from datetime import UTC, datetime

from pydantic import BaseModel

from convintel.contracts import Channel, Conversation, Speaker, Turn
from convintel.contracts.schemas import IntentOutput, OperatorResult
from convintel.errors import ConvIntelError
from convintel.operators.intent.operator import CLASSIFIER_SOURCE

DEFAULT_TEXT = "Where is my new card?"
FAKE_VERSION = "0.0.0-test"


def make_conversation(
    *texts: str, conversation_id: str = "conv-1", redacted: bool = True
) -> Conversation:
    """A chat conversation with one customer turn per text.

    Redacted by default: the graph must only ever receive redacted conversations.
    """
    return Conversation(
        id=conversation_id,
        channel=Channel.CHAT,
        started_at=datetime(2026, 10, 20, tzinfo=UTC),
        turns=tuple(
            Turn(index=i, speaker=Speaker.CUSTOMER, text=text)
            for i, text in enumerate(texts or (DEFAULT_TEXT,))
        ),
        redacted=redacted,
    )


def intent_output(label: str = "card_arrival", confidence: float = 0.95) -> IntentOutput:
    return IntentOutput(label=label, confidence=confidence, source=CLASSIFIER_SOURCE, top_k=[])


def as_result[T: BaseModel](name: str, output: T) -> OperatorResult[T]:
    return OperatorResult[T](
        operator=name, operator_version=FAKE_VERSION, output=output, latency_ms=1.0, model_id="fake"
    )


class FakeOperator[T: BaseModel]:
    """Satisfies the Operator protocol by shape: returns a fixed output, or raises."""

    version = FAKE_VERSION

    def __init__(self, name: str, output: T, *, error: ConvIntelError | None = None) -> None:
        self.name = name
        self._output = output
        self._error = error
        self.calls = 0

    async def run(self, conversation: Conversation) -> OperatorResult[T]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return as_result(self.name, self._output)


class FakeVerifier:
    """Records how often it is called; returns a fixed verification result."""

    def __init__(
        self, result: OperatorResult[IntentOutput] | None = None, errors: list[str] | None = None
    ) -> None:
        self._result = result
        self._errors = errors or []
        self.calls = 0

    async def __call__(
        self, conversation: Conversation, classified: OperatorResult[IntentOutput]
    ) -> tuple[OperatorResult[IntentOutput] | None, list[str]]:
        self.calls += 1
        return self._result, list(self._errors)
