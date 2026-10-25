from collections.abc import Callable
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from convintel.config import Settings
from convintel.llm import LLMClient, load_prompt
from convintel.llm.testing import FakeStructuredChatModel, no_sleep, tool_answer
from scripts.synth.judge import (
    JUDGE_PROMPT_VERSION,
    TAXONOMY,
    configured_client,
    evidence_errors,
    judge_input,
    judge_sample,
    reviewable,
    sent_temperature,
)
from scripts.synth.judge_check import DEFAULT_CONTROLS, Control, load_controls, run_gate
from scripts.synth.plausibility import INTENT_KIND
from scripts.synth.review import review_batch
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import (
    LABELS,
    Annotation,
    JudgeObservation,
    LabelTargets,
    ObservedFlag,
    ObservedIntent,
    ObservedSentiment,
    RunSpec,
    compare,
    targets_of,
)
from scripts.synth.storage import digest, load_samples
from scripts.synth.validation import Rejected


def run_spec(client: LLMClient) -> RunSpec:
    return RunSpec(
        seed=42,
        model_id=client.model_id,
        model_revision="test",
        judge_model_id=client.model_id,
        judge_model_revision="test",
        prompt_sha256=digest(load_prompt("synthetic", 5)),
        judge_prompt_sha256=digest(load_prompt("synthetic_judge", JUDGE_PROMPT_VERSION)),
        environment_sha256="b" * 64,
        pii_scan="patterns",
    )


def truthful(control: Control) -> JudgeObservation:
    """What a perfect judge would report for a control: its truth, with valid evidence."""
    last = len(control.turns) - 2
    t = control.truth
    return JudgeObservation(
        intent=ObservedIntent(value=t.intent, evidence_turns=(0,)),
        resolved=ObservedFlag(value=t.resolved, evidence_turns=(last + 1,)),
        escalation_requested=ObservedFlag(value=t.escalation_requested, evidence_turns=(0, 2)),
        start_sentiment=ObservedSentiment(value=t.start_sentiment, evidence_turns=(0,)),
        end_sentiment=ObservedSentiment(value=t.end_sentiment, evidence_turns=(last,)),
    )


async def test_blind_judge_never_sees_targets_or_pii_values(
    settings: Settings,
    sample_factory: Callable,
    judge_observation: Callable,
) -> None:
    sample = sample_factory(scenarios(42)[2])
    scenario = sample.labels.scenario
    model = FakeStructuredChatModel(
        responses=[tool_answer(JudgeObservation, judge_observation(scenario).model_dump())]
    )
    client = LLMClient(settings, chat_model=model)
    record = await judge_sample(client, sample, run_spec(client))
    prompt = " ".join(str(message.content) for message in model.calls[0])
    for value in sample.labels.pii_values:
        assert value.surface_form not in prompt
        assert value.canonical_value not in prompt
        assert value.slot.token in prompt
    assert scenario.model_dump_json() not in prompt
    assert '"start_sentiment":' not in prompt and '"escalation_requested":' not in prompt
    assert TAXONOMY in prompt and "<conversation>" in prompt
    assert record.input_sha256 == digest(judge_input(sample))
    assert record.passed and record.mismatched_labels == ()
    assert record.prompt_version == "synthetic_judge/v4"


@pytest.mark.parametrize("label", LABELS)
async def test_code_compares_observations_with_every_target(
    settings: Settings,
    sample_factory: Callable,
    judge_observation: Callable,
    label: str,
) -> None:
    sample = sample_factory(scenarios(42)[0])
    scenario = sample.labels.scenario
    observation = judge_observation(scenario)
    field = getattr(observation, label)
    wrong = {
        "intent": "exchange_rate",
        "resolved": not scenario.resolved,
        "escalation_requested": not scenario.escalation_requested,
        "start_sentiment": "positive" if scenario.start_sentiment != "positive" else "negative",
        "end_sentiment": "positive" if scenario.end_sentiment != "positive" else "negative",
    }[label]
    observation = observation.model_copy(
        update={label: type(field)(value=wrong, evidence_turns=field.evidence_turns)}
    )
    model = FakeStructuredChatModel(
        responses=[tool_answer(JudgeObservation, observation.model_dump())]
    )
    client = LLMClient(settings, chat_model=model)
    with pytest.raises(Rejected, match="label_mismatch") as caught:
        await judge_sample(client, sample, run_spec(client))
    assert caught.value.category == "judge"
    assert caught.value.judge is not None
    assert caught.value.judge.failed_labels == (label,)


def test_false_targets_match_false_observations() -> None:
    """The v1 failure mode: a false/negative target must agree with a false/negative reading."""
    control = next(
        c for c in load_controls(DEFAULT_CONTROLS) if c.id == "ctl-03-valid-informational"
    )
    observation = truthful(control)
    assert control.targets.resolved is False and control.targets.escalation_requested is False
    assert compare(observation, control.targets) == ()


@pytest.mark.parametrize(
    ("label", "turns", "reason"),
    [
        ("start_sentiment", (1,), "start_sentiment:agent_turn"),
        ("start_sentiment", (2,), "start_sentiment:missing_first_customer_turn"),
        ("end_sentiment", (0,), "end_sentiment:missing_last_customer_turn"),
        ("end_sentiment", (5,), "end_sentiment:agent_turn"),
        ("escalation_requested", (1,), "escalation_requested:agent_turn"),
        ("intent", (9,), "intent:out_of_range"),
        ("resolved", (6,), "resolved:out_of_range"),
    ],
)
def test_invalid_evidence_is_reported(label: str, turns: tuple[int, ...], reason: str) -> None:
    control = load_controls(DEFAULT_CONTROLS)[0]
    observation = truthful(control)
    field = getattr(observation, label)
    changed = observation.model_copy(
        update={label: type(field)(value=field.value, evidence_turns=turns)}
    )
    speakers = [turn.speaker for turn in control.turns]
    assert evidence_errors(observation, speakers) == ()
    assert evidence_errors(changed, speakers) == (reason,)


async def test_agent_sentiment_evidence_rejects_the_sample(
    settings: Settings,
    sample_factory: Callable,
    judge_observation: Callable,
) -> None:
    sample = sample_factory(scenarios(42)[0])
    observation = judge_observation(sample.labels.scenario)
    observation = observation.model_copy(
        update={
            "end_sentiment": ObservedSentiment(
                value=observation.end_sentiment.value, evidence_turns=(5,)
            )
        }
    )
    model = FakeStructuredChatModel(
        responses=[tool_answer(JudgeObservation, observation.model_dump())]
    )
    client = LLMClient(settings, chat_model=model)
    with pytest.raises(Rejected, match="invalid_evidence"):
        await judge_sample(client, sample, run_spec(client))


def test_observations_are_strict_and_always_cite_evidence() -> None:
    with pytest.raises(ValidationError):
        ObservedFlag.model_validate({"value": "true", "evidence_turns": [0]})
    for model in (ObservedFlag, ObservedIntent, ObservedSentiment):
        with pytest.raises(ValidationError):
            model.model_validate({"value": False, "evidence_turns": []})


async def test_closed_intent_schema_rejects_invented_labels(settings: Settings) -> None:
    controls = load_controls(DEFAULT_CONTROLS)
    invented = truthful(controls[0]).model_dump()
    invented["intent"]["value"] = "card_issue"
    model = FakeStructuredChatModel(
        responses=[
            tool_answer(JudgeObservation, invented),
            tool_answer(JudgeObservation, invented),
            *(tool_answer(JudgeObservation, truthful(c).model_dump()) for c in controls[1:]),
        ]
    )
    client = LLMClient(settings, chat_model=model, sleep=no_sleep)
    report = await run_gate(client, DEFAULT_CONTROLS)
    assert report.results[0].error == "LLMError" and not report.results[0].ok
    assert not report.gate_passed and report.passed_controls == "25/26"
    assert report.field_correct["intent"] == "25/26"


def test_controls_break_exactly_one_label_and_use_masked_pii() -> None:
    controls = load_controls(DEFAULT_CONTROLS)
    assert len(controls) == 26
    assert sum(c.broken is None for c in controls) == 11
    broken = [c.broken for c in controls if c.broken]
    assert set(broken) == set(LABELS)
    assert sum(c.targets.resolved is False for c in controls if c.broken == "resolved") == 3
    assert {
        c.targets.escalation_requested for c in controls if c.broken == "escalation_requested"
    } == {True, False}
    for control in controls:
        flagged = compare(truthful(control), control.targets)
        assert flagged == ((control.broken,) if control.broken else ())
        assert "@" not in " ".join(turn.text for turn in control.turns)


def test_control_targets_must_differ_only_in_the_broken_field() -> None:
    data = load_controls(DEFAULT_CONTROLS)[0].model_dump(mode="json")
    data["targets"]["resolved"] = not data["targets"]["resolved"]
    with pytest.raises(ValidationError, match="broken field"):
        Control.model_validate(data)


async def test_gate_passes_a_truthful_judge_and_reports_counts(
    settings: Settings, spans: InMemorySpanExporter
) -> None:
    controls = load_controls(DEFAULT_CONTROLS)
    model = FakeStructuredChatModel(
        responses=[tool_answer(JudgeObservation, truthful(c).model_dump()) for c in controls]
    )
    report = await run_gate(LLMClient(settings, chat_model=model), DEFAULT_CONTROLS)
    assert report.gate_passed and report.passed_controls == "26/26"
    assert report.field_correct == {label: "26/26" for label in LABELS}
    assert report.judge_model_id == "ollama:qwen3:8b" and report.cost_usd == 0
    assert report.judge_temperature == 0.0
    assert {s.name for s in spans.get_finished_spans()} >= {
        "synthetic.judge",
        "synthetic.judge_check",
    }
    assert all("text" not in str(s.attributes) for s in spans.get_finished_spans())


async def test_gate_fails_a_judge_that_reads_false_targets_as_mismatches(
    settings: Settings,
) -> None:
    controls = load_controls(DEFAULT_CONTROLS)

    def inverted(control: Control) -> JudgeObservation:
        observation = truthful(control)
        # Reproduces the pilot-v2 judge: a false escalation target read as "not observed".
        if not control.truth.escalation_requested:
            return observation.model_copy(
                update={"escalation_requested": ObservedFlag(value=True, evidence_turns=(0,))}
            )
        return observation

    model = FakeStructuredChatModel(
        responses=[tool_answer(JudgeObservation, inverted(c).model_dump()) for c in controls]
    )
    report = await run_gate(LLMClient(settings, chat_model=model), DEFAULT_CONTROLS)
    assert not report.gate_passed
    failed = {r.id for r in report.results if not r.ok}
    assert "ctl-01-valid-resolved" in failed and "ctl-08-broken-escalation-false" not in failed
    assert report.field_correct["escalation_requested"] == "4/26"
    assert report.field_correct["intent"] == "26/26"


def test_only_current_judge_passed_samples_reach_human_review(
    tmp_path: Path,
    sample_factory: Callable,
) -> None:
    sample = sample_factory(scenarios(42)[0])
    assert isinstance(sample.labels, Annotation) and sample.labels.judge is not None
    judge = sample.labels.judge
    absent = sample.model_copy(update={"labels": sample.labels.model_copy(update={"judge": None})})
    mismatched_observation = judge.observation.model_copy(
        update={"intent": ObservedIntent(value="exchange_rate", evidence_turns=(0,))}
    )
    # A record that claims to pass while its observation disagrees must still be refused.
    tampered = sample.model_copy(
        update={
            "labels": sample.labels.model_copy(
                update={"judge": judge.model_copy(update={"observation": mismatched_observation})}
            )
        }
    )
    turns = sample.conversation.turns
    changed = sample.model_copy(
        update={
            "conversation": sample.conversation.model_copy(
                update={
                    "turns": (
                        turns[0].model_copy(update={"text": "A completely different issue."}),
                        *turns[1:],
                    )
                }
            )
        }
    )
    assert reviewable(sample)
    assert compare(judge.observation, targets_of(sample.labels.scenario)) == ()
    for unqualified in (absent, tampered, changed):
        assert not reviewable(unqualified)

        def forbidden(_: str) -> str:
            raise AssertionError("Unqualified candidate reached human input")

        assert (
            review_batch(
                [unqualified],
                tmp_path / "review.jsonl",
                "human",
                ask=forbidden,
                display=lambda _: None,
            )
            == 0
        )
    assert not (tmp_path / "review.jsonl").exists()


def test_historical_candidates_remain_readable_but_ineligible() -> None:
    directory = Path(__file__).resolve().parents[2] / "data/synthetic/candidates/pilot-v1.1"
    samples = load_samples(directory)
    assert len(samples) == 10 and all(not reviewable(sample) for sample in samples)
    assert digest(samples[0]) == "49e26e9b0d90fc28c988d21bdc7380606bcd9f3e7b6b0c1d0c52913b8e0996a6"


def test_ollama_clients_disable_reasoning_without_network(settings: Settings) -> None:
    generator = configured_client(settings, "synthetic", 5)
    judge = configured_client(settings, "synthetic_judge", JUDGE_PROMPT_VERSION)
    assert generator.model_id == judge.model_id == "ollama:qwen3:8b"
    assert generator._chat_model is not None and judge._chat_model is not None
    assert getattr(generator._chat_model, "reasoning", None) is False
    assert getattr(judge._chat_model, "reasoning", None) is False
    assert getattr(generator._chat_model, "temperature", None) == 0.35
    assert getattr(judge._chat_model, "temperature", None) == 0


def test_label_targets_reject_non_boolean_flags() -> None:
    with pytest.raises(ValidationError):
        LabelTargets.model_validate(
            {
                "intent": "change_pin",
                "resolved": "false",
                "escalation_requested": False,
                "start_sentiment": "neutral",
                "end_sentiment": "neutral",
            }
        )


def test_gate_v2_keeps_the_original_controls_and_covers_the_failure_patterns() -> None:
    original = load_controls(DEFAULT_CONTROLS.parent / "judge_controls.jsonl")
    controls = load_controls(DEFAULT_CONTROLS.parent / "judge_controls_v2.jsonl")
    assert controls[:10] == original
    new = {c.id: c for c in controls[10:]}
    pairs = [
        ("ctl-11-valid-calm-problem", "ctl-12-broken-calm-problem-start"),
        ("ctl-13-valid-refund-processed", "ctl-14-broken-refund-processed"),
        ("ctl-15-valid-obstacle", "ctl-16-broken-obstacle"),
        ("ctl-17-valid-courtesy-close", "ctl-18-broken-courtesy-close"),
        ("ctl-19-valid-card-top-up-fee", "ctl-20-broken-pin-of-new-card"),
        ("ctl-21-valid-resolved-escalation", "ctl-22-broken-resolved-escalation"),
    ]
    assert set(new) == {name for pair in pairs for name in pair}
    for valid, broken in pairs:
        assert new[valid].broken is None and new[broken].broken is not None
    calm = new["ctl-11-valid-calm-problem"]
    assert INTENT_KIND[calm.truth.intent] == "problem" and calm.truth.start_sentiment == "neutral"
    assert not new["ctl-15-valid-obstacle"].truth.resolved
    escalation = new["ctl-21-valid-resolved-escalation"]
    assert escalation.truth.resolved and escalation.truth.escalation_requested
    assert (
        "supervisor" in escalation.turns[3].text and "joined the chat" in escalation.turns[3].text
    )


def test_gate_v3_keeps_gate_v2_verbatim_and_adds_borderline_sentiment_controls() -> None:
    v2 = (DEFAULT_CONTROLS.parent / "judge_controls_v2.jsonl").read_text().splitlines()
    v3 = DEFAULT_CONTROLS.read_text().splitlines()
    assert DEFAULT_CONTROLS.name == "judge_controls_v3.jsonl" and v3[:22] == v2
    new = load_controls(DEFAULT_CONTROLS)[22:]
    assert len(new) == 4
    assert all(c.broken in (None, "start_sentiment", "end_sentiment") for c in new)
    mild = [c for c in new if "negative" in (c.truth.start_sentiment, c.truth.end_sentiment)]
    calm = [c for c in new if c not in mild]
    assert len(mild) == len(calm) == 2
    assert {c.broken for c in mild} == {None, "end_sentiment"}
    assert {c.broken for c in calm} == {None, "start_sentiment"}
    assert all(c.truth.start_sentiment == "neutral" for c in calm)
    # Different wording from ctl-10 and the guide examples, so passing is not memorisation.
    text = " ".join(turn.text.lower() for c in new for turn in c.turns)
    assert not any(word in text for word in ("annoying", "frustrating", "worried"))


def test_judge_temperature_records_the_provider_default_for_sonnet() -> None:
    prompt = load_prompt("synthetic_judge", JUDGE_PROMPT_VERSION)
    for model, sent in (("claude-sonnet-5-5", None), ("claude-haiku-4-5", 0.0)):
        settings = Settings(_env_file=None, llm_provider="anthropic", llm_model=model)  # pyright: ignore[reportCallIssue]
        assert sent_temperature(LLMClient(settings), prompt) == sent
