import asyncio
import threading
import time
import traceback
from collections.abc import Sequence

import pytest
from sentiment_fakes import NEG, NEU, POS, FakeSentimentModel, make_conversation

from convintel.contracts import Operator, Sentiment, SentimentOutput, Speaker
from convintel.errors import OperatorError
from convintel.operators.sentiment import DEFAULT_MODEL_ID, SentimentOperator, SentimentScores
from convintel.operators.sentiment import operator as operator_module
from convintel.operators.sentiment.model import LabelMappingError

C, A = Speaker.CUSTOMER, Speaker.AGENT


async def test_scores_customer_turns_in_one_batch() -> None:
    fake = FakeSentimentModel({"this is broken": NEG, "ok": NEU, "thanks!": POS})
    conversation = make_conversation(
        (C, "this is broken"),
        (A, "I'm sorry to hear that"),
        (C, "ok"),
        (A, "Fixed it for you"),
        (C, "thanks!"),
    )

    result = await SentimentOperator(model=fake).run(conversation)

    assert fake.batches == [["this is broken", "ok", "thanks!"]]
    assert result.output.customer_trajectory == [
        Sentiment.NEGATIVE,
        Sentiment.NEUTRAL,
        Sentiment.POSITIVE,
    ]
    assert result.output.overall is Sentiment.POSITIVE


async def test_agent_and_bot_turns_are_ignored() -> None:
    fake = FakeSentimentModel({"still waiting": NEG})
    conversation = make_conversation(
        (Speaker.BOT, "Welcome!"), (C, "still waiting"), (A, "Happy to help!")
    )

    result = await SentimentOperator(model=fake).run(conversation)

    assert result.output.customer_trajectory == [Sentiment.NEGATIVE]
    assert result.output.overall is Sentiment.NEGATIVE


async def test_no_customer_turns_is_neutral_without_loading_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(model_id: str) -> None:
        raise AssertionError("model must not load")

    monkeypatch.setattr(operator_module, "get_sentiment_model", fail)

    result = await SentimentOperator().run(make_conversation((A, "Hello?")))

    assert result.output == SentimentOutput(overall=Sentiment.NEUTRAL, customer_trajectory=[])


async def test_result_envelope() -> None:
    operator = SentimentOperator(model=FakeSentimentModel({"hi": NEU}))

    result = await operator.run(make_conversation((C, "hi")))

    assert isinstance(operator, Operator)
    assert result.operator == "sentiment"
    assert result.operator_version == operator.version
    assert result.model_id == "fake/sentiment"
    assert result.latency_ms >= 0
    assert result.cost_usd is None


async def test_inference_runs_off_the_event_loop_thread() -> None:
    fake = FakeSentimentModel({"hi": NEU})

    await SentimentOperator(model=fake).run(make_conversation((C, "hi")))

    assert fake.threads
    assert threading.get_ident() not in fake.threads


async def test_uses_the_shared_model_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []
    fake = FakeSentimentModel({"hi": POS})

    def fake_get(model_id: str) -> FakeSentimentModel:
        requested.append(model_id)
        return fake

    monkeypatch.setattr(operator_module, "get_sentiment_model", fake_get)
    operator = SentimentOperator()

    result = await operator.run(make_conversation((C, "hi")))

    assert requested == [DEFAULT_MODEL_ID]
    assert result.model_id == DEFAULT_MODEL_ID


@pytest.mark.parametrize("error", [OSError("model not found"), LabelMappingError("LABEL_0")])
async def test_model_failures_become_operator_errors(error: Exception) -> None:
    class Broken(FakeSentimentModel):
        def predict(self, texts: Sequence[str]) -> list[SentimentScores]:
            raise error

    with pytest.raises(OperatorError) as excinfo:
        await SentimentOperator(model=Broken({})).run(make_conversation((C, "hi")))

    assert excinfo.value.operator == "sentiment"


async def test_errors_never_carry_turn_text() -> None:
    secret = "my card is <CREDIT_CARD> and I'm furious"

    class Leaky(FakeSentimentModel):
        def predict(self, texts: Sequence[str]) -> list[SentimentScores]:
            raise OSError(f"tokenizer failed on {texts[0]!r}")

    with pytest.raises(OperatorError) as excinfo:
        await SentimentOperator(model=Leaky({})).run(make_conversation((C, secret)))

    error = excinfo.value
    assert error.message == "sentiment inference failed (OSError)"
    assert error.__cause__ is None
    assert error.__suppress_context__
    assert secret not in "".join(traceback.format_exception(error))


async def test_concurrent_runs_are_serialized() -> None:
    """The shared pipeline is not thread-safe; to_thread alone would run these in parallel."""

    class Slow(FakeSentimentModel):
        active = 0
        peak = 0
        lock = threading.Lock()

        def predict(self, texts: Sequence[str]) -> list[SentimentScores]:
            with self.lock:
                Slow.active += 1
                Slow.peak = max(Slow.peak, Slow.active)
            time.sleep(0.02)
            with self.lock:
                Slow.active -= 1
            return super().predict(texts)

    model = Slow({"hi": NEU})
    operators = [SentimentOperator(model=model) for _ in range(6)]

    await asyncio.gather(*(op.run(make_conversation((C, "hi"))) for op in operators))

    assert len(model.batches) == 6
    assert Slow.peak == 1
