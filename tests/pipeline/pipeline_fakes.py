"""Shared test doubles for the pipeline tests."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from pydantic import BaseModel

from convintel.contracts import Channel, Conversation, Speaker, Turn
from convintel.contracts.schemas import IntentOutput, OperatorResult, PIIEntity, PIIOutput
from convintel.errors import ConvIntelError
from convintel.operators.intent.operator import CLASSIFIER_SOURCE

DEFAULT_TEXT = "Where is my new card?"
FAKE_VERSION = "0.0.0-test"
REDACTED_TEXT = "<EMAIL_ADDRESS>"
TAU = 0.861


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


def as_result[T: BaseModel](
    name: str, output: T, *, cost_usd: float | None = None, input_tokens: int | None = None
) -> OperatorResult[T]:
    return OperatorResult[T](
        operator=name,
        operator_version=FAKE_VERSION,
        output=output,
        latency_ms=1.0,
        model_id="fake",
        cost_usd=cost_usd,
        input_tokens=input_tokens,
    )


type WarmupHook = Callable[[], Awaitable[None]]


class FakeWarmup:
    """Counts warmup() calls; can fail, or await a hook first (concurrency tests)."""

    def __init__(
        self, *, error: ConvIntelError | None = None, hook: WarmupHook | None = None
    ) -> None:
        self._warmup_error = error
        self._warmup_hook = hook
        self.warmups = 0

    async def warmup(self) -> None:
        self.warmups += 1
        if self._warmup_hook is not None:
            await self._warmup_hook()
        if self._warmup_error is not None:
            raise self._warmup_error


class FakeOperator[T: BaseModel](FakeWarmup):
    """Satisfies WarmableOperator by shape: records what it saw, returns a fixed output."""

    version = FAKE_VERSION

    def __init__(
        self,
        name: str,
        output: T,
        *,
        error: ConvIntelError | None = None,
        cost_usd: float | None = None,
        input_tokens: int | None = None,
        warmup_error: ConvIntelError | None = None,
        warmup_hook: WarmupHook | None = None,
    ) -> None:
        super().__init__(error=warmup_error, hook=warmup_hook)
        self.name = name
        self._output = output
        self._error = error
        self._cost_usd = cost_usd
        self._input_tokens = input_tokens
        self.seen: list[Conversation] = []

    @property
    def calls(self) -> int:
        return len(self.seen)

    async def run(self, conversation: Conversation) -> OperatorResult[T]:
        self.seen.append(conversation)
        if self._error is not None:
            raise self._error
        return as_result(
            self.name, self._output, cost_usd=self._cost_usd, input_tokens=self._input_tokens
        )


class FakeIntentClassifier(FakeOperator[IntentOutput]):
    """Intent operator double with the τ shipped by its model bundle, known after warmup."""

    def __init__(
        self,
        output: IntentOutput,
        *,
        threshold: float | None = TAU,
        warmup_error: ConvIntelError | None = None,
        warmup_hook: WarmupHook | None = None,
    ) -> None:
        super().__init__("intent", output, warmup_error=warmup_error, warmup_hook=warmup_hook)
        self._threshold = threshold
        self.warmed = False

    @property
    def threshold(self) -> float | None:
        return self._threshold if self.warmed else None

    async def warmup(self) -> None:
        await super().warmup()
        self.warmed = True


class FakeRedactor(FakeWarmup):
    """Replaces every turn's text; can fail, or 'forget' to set the redacted flag."""

    name = "pii"
    version = FAKE_VERSION

    def __init__(
        self,
        *,
        error: Exception | None = None,
        sets_flag: bool = True,
        warmup_error: ConvIntelError | None = None,
        warmup_hook: WarmupHook | None = None,
    ) -> None:
        super().__init__(error=warmup_error, hook=warmup_hook)
        self._error = error
        self._sets_flag = sets_flag
        self.calls = 0

    async def run(self, conversation: Conversation) -> OperatorResult[PIIOutput]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        redacted = conversation.model_copy(
            update={
                "turns": tuple(
                    t.model_copy(update={"text": REDACTED_TEXT}) for t in conversation.turns
                ),
                "redacted": self._sets_flag,
            }
        )
        entity = PIIEntity(entity_type="EMAIL_ADDRESS", turn_index=0, start=0, end=1, score=1.0)
        return as_result(self.name, PIIOutput(entities=[entity], redacted_conversation=redacted))


class FakeVerifier(FakeWarmup):
    """Records how often it is called; returns a fixed verification result."""

    version = FAKE_VERSION

    def __init__(
        self,
        result: OperatorResult[IntentOutput] | None = None,
        errors: list[str] | None = None,
        *,
        warmup_error: ConvIntelError | None = None,
        warmup_hook: WarmupHook | None = None,
    ) -> None:
        super().__init__(error=warmup_error, hook=warmup_hook)
        self._result = result
        self._errors = errors or []
        self.calls = 0

    async def __call__(
        self, conversation: Conversation, classified: OperatorResult[IntentOutput]
    ) -> tuple[OperatorResult[IntentOutput] | None, list[str]]:
        self.calls += 1
        return self._result, list(self._errors)
