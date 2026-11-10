import json
import logging
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from prometheus_client import REGISTRY
from pydantic import SecretStr, ValidationError

from convintel.config import Settings, get_settings
from convintel.contracts import (
    Channel,
    Conversation,
    ConversationInsights,
    IntentOutput,
    OperatorResult,
    Speaker,
    Turn,
)
from convintel.errors import OperatorError
from convintel.llm import LLMClient
from convintel.llm.testing import FakeStructuredChatModel, no_sleep, tool_answer
from convintel.operators.intent.descriptions import intent_descriptions
from convintel.operators.intent.operator import CLASSIFIER_SOURCE, UNKNOWN_LABEL
from convintel.pipeline.verifier import (
    LETTERS,
    LLMIntentVerifier,
    VerifierAnswer,
    answer_model,
    label_for,
    render_candidates,
    shuffled_candidates,
)

V3 = intent_descriptions(3)
REASONING = "SENTINEL-REASONING-a1b2c3 because the customer said so"
# The classifier's ranking, top-1 first, with calibrated probabilities.
TOP_K: list[tuple[str, float]] = [
    ("card_arrival", 0.52),
    ("card_delivery_estimate", 0.21),
    ("lost_or_stolen_card", 0.12),
    ("order_physical_card", 0.08),
    ("get_physical_card", 0.04),
]
CONVERSATION_ID = "conv-verifier-001"
# Pinned: the order for CONVERSATION_ID. A different seed derivation (Python's hash(), another
# digest, other bytes) changes it, and so would showing the candidates in the classifier's order.
SHUFFLED = [
    "card_delivery_estimate",
    "order_physical_card",
    "get_physical_card",
    "lost_or_stolen_card",
    "card_arrival",
]

# Synthetic and already redacted, as the pipeline hands it over: placeholders from the PII step.
CONVERSATION = Conversation(
    id=CONVERSATION_ID,
    channel=Channel.CHAT,
    started_at=datetime(2026, 11, 1, tzinfo=UTC),
    redacted=True,
    turns=(
        Turn(index=0, speaker=Speaker.BOT, text="Hello! How can I help you today?"),
        Turn(index=1, speaker=Speaker.CUSTOMER, text="I'm <PERSON>, my card never came."),
        Turn(index=2, speaker=Speaker.AGENT, text="I can check, is <EMAIL_ADDRESS> your email?"),
        Turn(index=3, speaker=Speaker.CUSTOMER, text="Yes. It was sent two weeks ago."),
    ),
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_API_KEY", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_provider": "anthropic",
        "llm_model": "claude-haiku-4-5",
        "anthropic_api_key": SecretStr("test-key"),
        **overrides,
    }
    return Settings(_env_file=None, **values)  # pyright: ignore[reportCallIssue]


def _classified(top_k: list[tuple[str, float]] = TOP_K) -> OperatorResult[IntentOutput]:
    output = IntentOutput(
        label=top_k[0][0], confidence=top_k[0][1], source=CLASSIFIER_SOURCE, top_k=top_k
    )
    return OperatorResult[IntentOutput](
        operator="intent", operator_version="1.0.0", output=output, latency_ms=40.0, model_id="m"
    )


def _answer(choice: str, n: int = 5) -> Any:
    return tool_answer(
        answer_model(n), {"reasoning": REASONING, "choice": choice}, input_tokens=900
    )


async def _verifier(*responses: Any) -> tuple[LLMIntentVerifier, FakeStructuredChatModel]:
    model = FakeStructuredChatModel(responses=list(responses))
    verifier = LLMIntentVerifier(LLMClient(_settings(), chat_model=model, sleep=no_sleep))
    await verifier.warmup()
    return verifier, model


async def _verify(
    verifier: LLMIntentVerifier,
    classified: OperatorResult[IntentOutput] | None = None,
    conversation: Conversation = CONVERSATION,
) -> tuple[OperatorResult[IntentOutput] | None, list[str]]:
    # The pipeline opens this span with traced(); the verifier only annotates it.
    with trace.get_tracer("test").start_as_current_span("operator.intent_verifier"):
        return await verifier(conversation, classified or _classified())


def _outcome_count(outcome: str) -> float:
    sample = REGISTRY.get_sample_value(
        "convintel_intent_verifier_invocations_total", {"outcome": outcome}
    )
    return sample or 0.0


def _span(spans: InMemorySpanExporter, name: str = "operator.intent_verifier") -> Any:
    (span,) = [s for s in spans.get_finished_spans() if s.name == name]
    return span


# --- answer schema ----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3, 5])
def test_the_answer_accepts_only_the_letters_with_a_candidate_and_none(n: int) -> None:
    model = answer_model(n)
    for letter in (*LETTERS[:n], "NONE"):
        assert model(reasoning="r", choice=letter).choice == letter
    for letter in LETTERS[n:]:
        with pytest.raises(ValidationError):
            model(reasoning="r", choice=letter)


def test_reasoning_comes_before_the_choice() -> None:
    # The model writes the fields in schema order: it reasons before it commits to a letter.
    assert list(answer_model(5).model_json_schema()["properties"]) == ["reasoning", "choice"]


def test_there_is_one_answer_class_per_candidate_count() -> None:
    assert answer_model(3) is answer_model(3)
    assert answer_model(3) is not answer_model(4)
    assert issubclass(answer_model(3), VerifierAnswer)


@pytest.mark.parametrize("n", [0, 6])
def test_the_answer_needs_one_to_five_candidates(n: int) -> None:
    with pytest.raises(ValueError, match="1 to 5"):
        answer_model(n)


# --- candidate order ------------------------------------------------------------------------


def test_the_order_is_fixed_by_the_conversation_id() -> None:
    labels = [label for label, _ in TOP_K]
    assert shuffled_candidates(CONVERSATION_ID, labels) == SHUFFLED
    assert shuffled_candidates(CONVERSATION_ID, labels) == SHUFFLED  # same id, same order


def test_the_order_is_not_the_classifier_ranking() -> None:
    labels = [label for label, _ in TOP_K]
    shuffled = shuffled_candidates(CONVERSATION_ID, labels)
    assert shuffled != labels
    assert sorted(shuffled) == sorted(labels)


def test_the_classifier_top1_has_no_favourite_letter() -> None:
    # Position bias: across many conversations, the classifier's top-1 must not cluster on a
    # letter (e.g. always A). 1,000 fixed ids keep this deterministic; uniform is 20 % each.
    labels = [label for label, _ in TOP_K]
    positions = Counter(
        shuffled_candidates(f"conv-{i:04d}", labels).index(labels[0]) for i in range(1000)
    )
    assert set(positions) == set(range(5))
    for count in positions.values():
        assert 150 <= count <= 250


@pytest.mark.parametrize("n", [0, 6])
def test_shuffling_needs_one_to_five_candidates(n: int) -> None:
    with pytest.raises(ValueError, match="1 to 5"):
        shuffled_candidates("c", [f"label_{i}" for i in range(n)])


def test_candidates_render_as_letters_and_descriptions_only() -> None:
    labels = ["card_arrival", "change_pin"]
    rendered = render_candidates(labels, V3)
    assert rendered == f"A. {V3['card_arrival']}\nB. {V3['change_pin']}"
    assert "card_arrival" not in rendered
    assert "change_pin" not in rendered


@pytest.mark.parametrize(("choice", "expected"), [("A", "x"), ("C", "z"), ("NONE", None)])
def test_a_letter_maps_back_to_its_label(choice: str, expected: str | None) -> None:
    assert label_for(choice, ["x", "y", "z"]) == expected


def test_a_letter_without_a_candidate_is_a_bug() -> None:
    with pytest.raises(ValueError, match="no candidate"):
        label_for("D", ["x", "y", "z"])


# --- what reaches the LLM ---------------------------------------------------------------------


async def test_the_prompt_shows_the_redacted_conversation_and_letters_only() -> None:
    verifier, model = await _verifier(_answer("B"))
    await _verify(verifier)

    (messages,) = model.calls
    system, user = (str(m.content) for m in messages)
    # The full redacted conversation, every speaker, wrapped as data.
    assert "<conversation>" in user
    assert "<PERSON>" in user
    assert "<EMAIL_ADDRESS>" in user
    assert "Hello! How can I help you today?" in user
    assert "customer's reason for contacting" in system
    # Each candidate as its letter and pinned v3 description, in the shuffled order.
    for letter, label in zip(LETTERS, SHUFFLED, strict=True):
        assert f"{letter}. {V3[label]}" in user
    # Never the label names, the probabilities or the classifier's preference.
    for label, probability in TOP_K:
        assert label not in system + user
        assert str(probability) not in system + user
    assert "classifier" not in (system + user).lower()


async def test_fewer_candidates_show_only_their_letters() -> None:
    top_k = TOP_K[:3]
    verifier, model = await _verifier(_answer("A", n=3))
    await _verify(verifier, _classified(top_k))

    user = str(model.calls[0][1].content)
    labels = shuffled_candidates(CONVERSATION_ID, [label for label, _ in top_k])
    assert render_candidates(labels, V3) in user
    assert "\nD. " not in user
    assert "\nE. " not in user


# --- outcomes ---------------------------------------------------------------------------------


async def test_choosing_the_classifier_top1_confirms_it(spans: InMemorySpanExporter) -> None:
    before = _outcome_count("confirmed")
    letter = LETTERS[SHUFFLED.index("card_arrival")]
    verifier, _ = await _verifier(_answer(letter))

    verified, errors = await _verify(verifier)

    assert errors == []
    assert verified is not None
    assert verified.output == IntentOutput(
        label="card_arrival", confidence=0.52, source="llm_verifier", top_k=TOP_K
    )
    assert _outcome_count("confirmed") - before == 1
    assert _span(spans).attributes["verifier.outcome"] == "confirmed"


async def test_choosing_another_candidate_overrides_with_its_probability(
    spans: InMemorySpanExporter,
) -> None:
    before = _outcome_count("overridden")
    letter = LETTERS[SHUFFLED.index("lost_or_stolen_card")]
    verifier, _ = await _verifier(_answer(letter))

    verified, errors = await _verify(verifier)

    assert errors == []
    assert verified is not None
    assert verified.output == IntentOutput(
        label="lost_or_stolen_card", confidence=0.12, source="llm_verifier", top_k=TOP_K
    )
    assert _outcome_count("overridden") - before == 1
    assert _span(spans).attributes["verifier.outcome"] == "overridden"


async def test_none_rejects_every_candidate(spans: InMemorySpanExporter) -> None:
    before = _outcome_count("rejected")
    verifier, _ = await _verifier(_answer("NONE"))

    verified, errors = await _verify(verifier)

    assert errors == []
    assert verified is not None
    assert verified.output == IntentOutput(
        label=UNKNOWN_LABEL, confidence=0.0, source="llm_verifier", top_k=TOP_K
    )
    assert _outcome_count("rejected") - before == 1
    assert _span(spans).attributes["verifier.outcome"] == "rejected"


async def test_the_result_fills_the_intent_slot_with_the_verifier_accounting(
    spans: InMemorySpanExporter,
) -> None:
    verifier, _ = await _verifier(_answer("A"))

    verified, _ = await _verify(verifier)

    assert verified is not None
    assert verified.operator == "intent"
    assert verified.operator_version == "1.0.0"
    assert verified.latency_ms > 40.0  # the classifier's 40 ms plus the verifier's own
    assert verified.model_id == "anthropic:claude-haiku-4-5"
    assert verified.prompt_version == "intent_verifier/v1"
    assert verified.input_tokens == 900
    assert verified.output_tokens == 20
    assert verified.cost_usd == pytest.approx(900 / 1e6 * 1 + 20 / 1e6 * 5)
    attributes = _span(spans).attributes
    assert attributes["conversation.id"] == CONVERSATION_ID
    assert attributes["verifier.candidates"] == 5
    assert attributes["prompt.version"] == "intent_verifier/v1"
    assert attributes["model.id"] == "anthropic:claude-haiku-4-5"
    assert attributes["llm.tokens.input"] == 900
    assert attributes["llm.tokens.output"] == 20
    assert attributes["llm.cost_usd"] == pytest.approx(verified.cost_usd)


async def test_a_letter_without_a_candidate_gets_one_repair() -> None:
    top_k = TOP_K[:3]
    verifier, model = await _verifier(_answer("E", n=5), _answer("A", n=3))

    verified, errors = await _verify(verifier, _classified(top_k))

    assert errors == []
    assert verified is not None
    assert len(model.calls) == 2  # the original call and exactly one repair


async def test_an_unrepaired_answer_fails_and_keeps_the_classifier(
    spans: InMemorySpanExporter,
) -> None:
    before = _outcome_count("failed")
    verifier, model = await _verifier(_answer("E"), _answer("E"))

    verified, errors = await _verify(verifier, _classified(TOP_K[:3]))

    assert verified is None
    assert errors == ["intent_verifier:LLMError"]
    assert len(model.calls) == 2
    assert _outcome_count("failed") - before == 1
    span = _span(spans)
    assert span.attributes["verifier.outcome"] == "failed"
    assert span.attributes["error.type"] == "LLMError"
    assert span.status.status_code is StatusCode.ERROR


async def test_a_bug_propagates() -> None:
    verifier, _ = await _verifier(RuntimeError("bug"))
    with pytest.raises(RuntimeError, match="bug"):
        await _verify(verifier)


async def test_the_reasoning_never_leaves_the_verifier(
    spans: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    # A failed then a successful verification: the repair path logs, the failure path errors.
    failing, _ = await _verifier(_answer("E"), _answer("E"))
    _, errors = await _verify(failing, _classified(TOP_K[:3]))
    working, _ = await _verifier(_answer("B"))
    verified, _ = await _verify(working)

    insights = ConversationInsights(
        conversation_id=CONVERSATION_ID, pipeline_version="test", intent=verified, errors=errors
    )
    exported = json.dumps(
        [dict(s.attributes or {}) | {"name": s.name} for s in spans.get_finished_spans()]
    )
    for text in (exported, caplog.text, insights.model_dump_json(), repr(errors)):
        assert "SENTINEL-REASONING" not in text


# --- warmup -----------------------------------------------------------------------------------


async def test_warmup_loads_the_prompt_without_calling_the_llm(
    spans: InMemorySpanExporter,
) -> None:
    model = FakeStructuredChatModel(responses=[])
    verifier = LLMIntentVerifier(LLMClient(_settings(), chat_model=model))

    await verifier.warmup()

    assert model.calls == []
    assert _span(spans, "operator.intent_verifier.warmup").attributes["prompt.version"] == (
        "intent_verifier/v1"
    )


async def test_a_missing_api_key_fails_warmup(spans: InMemorySpanExporter) -> None:
    verifier = LLMIntentVerifier(LLMClient(_settings(anthropic_api_key=None)))

    with pytest.raises(OperatorError, match="ANTHROPIC_API_KEY") as excinfo:
        await verifier.warmup()

    assert excinfo.value.operator == "intent_verifier"
    span = _span(spans, "operator.intent_verifier.warmup")
    assert span.status.status_code is StatusCode.ERROR


async def test_verifying_before_warmup_is_a_bug() -> None:
    verifier = LLMIntentVerifier(
        LLMClient(_settings(), chat_model=FakeStructuredChatModel(responses=[]))
    )
    with pytest.raises(RuntimeError, match="before warmup"):
        await verifier(CONVERSATION, _classified())
