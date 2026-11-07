import asyncio
import threading
import time
import traceback
from collections.abc import Iterator, Sequence
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from sentiment_fakes import (
    NEG,
    NEU,
    POS,
    FakeSentimentModel,
    RecordingTracer,
    make_conversation,
)

from convintel.contracts import Operator, Sentiment, SentimentOutput, Speaker
from convintel.errors import OperatorError
from convintel.operators.sentiment import DEFAULT_MODEL_ID, SentimentOperator, SentimentScores
from convintel.operators.sentiment import operator as operator_module
from convintel.operators.sentiment.model import LabelMappingError

C, A = Speaker.CUSTOMER, Speaker.AGENT


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    yield exporter
    provider.shutdown()


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


async def test_warmup_keeps_the_provisioning_message_of_a_missing_model(
    monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    missing = OperatorError("sentiment", "model is missing; provision it with: <command>")

    def fail(model_id: str) -> FakeSentimentModel:
        raise missing

    monkeypatch.setattr(operator_module, "get_sentiment_model", fail)

    with pytest.raises(OperatorError) as excinfo:
        await SentimentOperator().warmup()

    assert excinfo.value is missing


@pytest.mark.parametrize("error", [OSError("model not found"), LabelMappingError("LABEL_0")])
async def test_model_failures_become_operator_errors(error: Exception) -> None:
    class Broken(FakeSentimentModel):
        def predict(self, texts: Sequence[str]) -> list[SentimentScores]:
            raise error

    with pytest.raises(OperatorError) as excinfo:
        await SentimentOperator(model=Broken({})).run(make_conversation((C, "hi")))

    assert excinfo.value.operator == "sentiment"


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, ValueError, LabelMappingError])
async def test_errors_never_carry_turn_text(error_type: type[Exception]) -> None:
    """Mirror of the PII test: a redaction miss must not leak through an error either."""
    secret = "call Marisol Quintavalle on 415-555-0123"

    class Leaky(FakeSentimentModel):
        def predict(self, texts: Sequence[str]) -> list[SentimentScores]:
            raise error_type(f"pipeline failed on {texts[0]!r}")

    tracer = RecordingTracer()  # stands in for the span the pipeline opens around run()

    with (
        pytest.raises(OperatorError) as excinfo,
        tracer.start_as_current_span("operator.sentiment"),
    ):
        await SentimentOperator(model=Leaky({})).run(make_conversation((C, secret)))

    error = excinfo.value
    assert error.message == f"sentiment inference failed ({error_type.__name__})"
    assert error.__cause__ is None
    assert error.__suppress_context__
    assert secret not in "".join(traceback.format_exception(error))
    assert len(tracer.recorded) == 1
    assert secret not in tracer.recorded[0]


async def test_run_annotates_the_callers_span() -> None:
    # The pipeline opens operator.sentiment with traced(); a span opened inside run() would
    # duplicate it, and these attributes would land on the inner span instead.
    secret = "my card was charged twice and nobody answers"
    parent = MagicMock(spec=trace.Span)
    with trace.use_span(parent, end_on_exit=False):
        result = await SentimentOperator(model=FakeSentimentModel({secret: NEG})).run(
            make_conversation((C, secret), (A, "let me check"))
        )

    attributes = {c.args[0]: c.args[1] for c in parent.set_attribute.call_args_list}
    assert attributes["sentiment.customer_turns"] == 1
    assert attributes["sentiment.overall"] == result.output.overall.value
    assert attributes["model.id"] == FakeSentimentModel.model_id
    assert "operator.version" not in attributes  # traced() sets it, once
    assert secret not in repr(attributes)


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


async def test_warmup_loads_the_model_once_and_run_reuses_it(
    monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    warmup_text = operator_module._WARMUP_TEXT  # pyright: ignore[reportPrivateUsage]
    fake = FakeSentimentModel({warmup_text: NEU, "hi": POS})
    requested: list[str] = []

    def load_once(model_id: str) -> FakeSentimentModel:
        requested.append(model_id)
        return fake

    def never_load(model_id: str) -> FakeSentimentModel:
        raise AssertionError("run() after warmup() must reuse the warmed model")

    monkeypatch.setattr(operator_module, "get_sentiment_model", load_once)
    operator = SentimentOperator()

    await operator.warmup()
    monkeypatch.setattr(operator_module, "get_sentiment_model", never_load)
    result = await operator.run(make_conversation((C, "hi")))

    assert requested == [DEFAULT_MODEL_ID]
    assert fake.batches == [[warmup_text], ["hi"]]
    assert threading.get_ident() not in fake.threads
    assert result.output.overall is Sentiment.POSITIVE
    (span,) = [s for s in spans.get_finished_spans() if s.name == "operator.sentiment.warmup"]
    attributes = span.attributes or {}
    assert attributes["model.id"] == DEFAULT_MODEL_ID
    assert attributes["operator.version"] == operator.version
    assert span.status.status_code is not StatusCode.ERROR


@pytest.mark.parametrize("error", [OSError("model not found"), LabelMappingError("LABEL_0")])
async def test_warmup_failure_raises_operator_error_and_marks_the_span(
    error: Exception, monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    def fail(model_id: str) -> FakeSentimentModel:
        raise error

    monkeypatch.setattr(operator_module, "get_sentiment_model", fail)

    with pytest.raises(OperatorError) as excinfo:
        await SentimentOperator().warmup()

    assert excinfo.value.operator == "sentiment"
    assert excinfo.value.message == f"warmup failed ({type(error).__name__})"
    assert excinfo.value.__cause__ is None
    (span,) = [s for s in spans.get_finished_spans() if s.name == "operator.sentiment.warmup"]
    assert span.status.status_code is StatusCode.ERROR
    assert (span.attributes or {})["error.type"] == "OperatorError"
    assert not span.events
