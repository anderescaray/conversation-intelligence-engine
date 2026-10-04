import asyncio
import threading
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from intent_fakes import LABELS, FailingIntentModel, FakeIntentModel, make_conversation
from numpy.typing import NDArray
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from convintel.contracts import Operator, Speaker
from convintel.errors import OperatorError
from convintel.operators.intent import (
    UNKNOWN_LABEL,
    IntentModel,
    IntentModelUnavailableError,
    IntentOperator,
    pick_intent,
    softmax,
)
from convintel.operators.intent import operator as operator_module

C, A = Speaker.CUSTOMER, Speaker.AGENT

GREETING = [0.2, 0.1, 0.0, 0.1]
LOST_CARD = [0.0, 5.0, 0.5, 0.0]
WHERE_IS_CARD = [3.0, 1.0, 0.0, 0.0]


def _never_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(uri: str, cache_dir: Path, device: str | None) -> IntentModel:
        raise AssertionError("model must not load")

    monkeypatch.setattr(operator_module, "get_intent_model", fail)


async def test_classifies_customer_turns_in_one_batch_and_keeps_the_clearest() -> None:
    fake = FakeIntentModel({"hi there": GREETING, "my card is gone": LOST_CARD})
    conversation = make_conversation(
        (C, "hi there"), (A, "Hello, how can I help?"), (C, "my card is gone")
    )

    result = await IntentOperator(model=fake).run(conversation)

    assert fake.batches == [["hi there", "my card is gone"]]
    assert result.output.label == "lost_or_stolen_card"
    assert result.output.source == "classifier"


async def test_confidence_and_top_k_are_temperature_calibrated() -> None:
    fake = FakeIntentModel({"where is my card": WHERE_IS_CARD}, temperature=2.0)

    result = await IntentOperator(model=fake, top_k=2).run(
        make_conversation((C, "where is my card"))
    )

    expected = softmax(np.array([WHERE_IS_CARD]), 2.0)[0]
    assert result.output.confidence == pytest.approx(expected[0])
    assert result.output.confidence < softmax(np.array([WHERE_IS_CARD]))[0][0]
    assert [label for label, _ in result.output.top_k] == ["card_arrival", "lost_or_stolen_card"]
    assert [p for _, p in result.output.top_k] == pytest.approx([expected[0], expected[1]])


def test_pick_intent_ranks_top_k_in_descending_order() -> None:
    probs = np.array([[0.1, 0.2, 0.3, 0.4], [0.05, 0.9, 0.03, 0.02]])

    output = pick_intent(probs, LABELS, top_k=3)

    assert output.label == "lost_or_stolen_card"
    assert output.confidence == pytest.approx(0.9)
    assert [label for label, _ in output.top_k] == [
        "lost_or_stolen_card",
        "card_arrival",
        "exchange_rate",
    ]


def test_top_k_larger_than_the_label_set_returns_every_label() -> None:
    output = pick_intent(np.array([[0.1, 0.2, 0.3, 0.4]]), LABELS, top_k=10)

    assert len(output.top_k) == len(LABELS)


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="top_k"):
        IntentOperator(model=FakeIntentModel({}), top_k=0)


async def test_no_customer_turns_is_unknown_without_loading_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _never_load(monkeypatch)

    result = await IntentOperator().run(make_conversation((A, "Hello?"), (Speaker.BOT, "Hi!")))

    assert result.output.label == UNKNOWN_LABEL
    assert result.output.confidence == 0.0
    assert result.output.top_k == []
    assert result.model_id is None


async def test_result_envelope() -> None:
    operator = IntentOperator(model=FakeIntentModel({"x": LOST_CARD}))

    result = await operator.run(make_conversation((C, "x")))

    assert isinstance(operator, Operator)
    assert result.operator == "intent"
    assert result.operator_version == operator.version
    assert result.model_id == "fake/intent"
    assert result.latency_ms >= 0
    assert result.cost_usd is None


async def test_inference_runs_off_the_event_loop_thread() -> None:
    fake = FakeIntentModel({"x": LOST_CARD})

    await IntentOperator(model=fake).run(make_conversation((C, "x")))

    assert fake.threads[0] != threading.get_ident()


async def test_concurrent_runs_never_overlap_inside_the_model() -> None:
    class SlowModel(FakeIntentModel):
        def __init__(self) -> None:
            super().__init__({"x": LOST_CARD})
            self.active = 0
            self.max_active = 0
            self._guard = threading.Lock()

        def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float32]:
            with self._guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self._guard:
                self.active -= 1
            return super().predict_logits(texts)

    model = SlowModel()
    operators = [IntentOperator(model=model) for _ in range(4)]

    await asyncio.gather(*(op.run(make_conversation((C, "x"))) for op in operators))

    assert model.max_active == 1


async def test_inference_errors_keep_only_the_exception_type() -> None:
    secret = "customer said 4111 1111 1111 1111"
    operator = IntentOperator(model=FailingIntentModel(ValueError(secret)))

    with pytest.raises(OperatorError) as info:
        await operator.run(make_conversation((C, "x")))

    assert info.value.operator == "intent"
    assert str(info.value) == "[intent] intent inference failed (ValueError)"
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__
    assert secret not in "".join(traceback.format_exception(info.value))


async def test_a_missing_model_surfaces_its_explicit_message() -> None:
    error = IntentModelUnavailableError("intent model bundle not found at /models/x")
    operator = IntentOperator(model=FailingIntentModel(error))

    with pytest.raises(OperatorError, match="bundle not found at /models/x"):
        await operator.run(make_conversation((C, "x")))


async def test_span_has_metadata_but_no_text(spans: InMemorySpanExporter) -> None:
    text = "my card is gone"
    fake = FakeIntentModel({text: LOST_CARD})

    await IntentOperator(model=fake).run(make_conversation((C, text), conversation_id="c-42"))

    (span,) = [s for s in spans.get_finished_spans() if s.name == "operator.intent"]
    attributes = dict(span.attributes or {})
    assert attributes["conversation.id"] == "c-42"
    assert attributes["model.id"] == "fake/intent"
    assert attributes["intent.customer_turns"] == 1
    assert attributes["intent.label"] == "lost_or_stolen_card"
    assert 0 < float(attributes["intent.confidence"]) <= 1  # pyright: ignore[reportArgumentType]
    assert text not in repr(attributes)


async def test_failed_span_records_the_type_only(spans: InMemorySpanExporter) -> None:
    operator = IntentOperator(model=FailingIntentModel(RuntimeError("raw text here")))

    with pytest.raises(OperatorError):
        await operator.run(make_conversation((C, "x")))

    (span,) = [s for s in spans.get_finished_spans() if s.name == "operator.intent"]
    assert span.status.status_code is StatusCode.ERROR
    assert (span.attributes or {})["error.type"] == "OperatorError"
    assert not span.events


async def test_warmup_loads_the_model_once_and_runs_one_inference(
    monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    fake = FakeIntentModel({operator_module._WARMUP_TEXT: GREETING, "x": LOST_CARD})  # pyright: ignore[reportPrivateUsage]
    loads: list[str] = []

    def load(uri: str, cache_dir: Path, device: str | None) -> IntentModel:
        loads.append(uri)
        return fake

    monkeypatch.setattr(operator_module, "get_intent_model", load)
    operator = IntentOperator(model_uri="models:/intent-classifier@champion")
    assert operator.threshold is None

    await operator.warmup()
    await operator.run(make_conversation((C, "x")))

    assert loads == ["models:/intent-classifier@champion"]
    assert len(fake.batches) == 2
    assert operator.threshold == 0.9
    (span,) = [s for s in spans.get_finished_spans() if s.name == "operator.intent.warmup"]
    assert (span.attributes or {})["model.id"] == "fake/intent"


async def test_warmup_failure_raises_operator_error_and_marks_the_span(
    monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    def load(uri: str, cache_dir: Path, device: str | None) -> IntentModel:
        raise IntentModelUnavailableError("cannot resolve models:/x@champion (MlflowException)")

    monkeypatch.setattr(operator_module, "get_intent_model", load)

    with pytest.raises(OperatorError, match="cannot resolve"):
        await IntentOperator(model_uri="models:/x@champion").warmup()

    (span,) = [s for s in spans.get_finished_spans() if s.name == "operator.intent.warmup"]
    assert span.status.status_code is StatusCode.ERROR
    assert not span.events


async def test_run_loads_lazily_without_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeIntentModel({"x": LOST_CARD})
    monkeypatch.setattr(operator_module, "get_intent_model", lambda uri, cache_dir, device: fake)

    result = await IntentOperator().run(make_conversation((C, "x")))

    assert result.output.label == "lost_or_stolen_card"
