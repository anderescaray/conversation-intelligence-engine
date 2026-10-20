from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from convintel.config import Settings
from convintel.contracts import Sentiment
from convintel.errors import ConvIntelError
from convintel.llm import LLMClient
from convintel.llm.testing import FakeStructuredChatModel, tool_answer
from scripts.synth.generate import generate_run
from scripts.synth.pii import values_for
from scripts.synth.plausibility import (
    INTENT_KIND,
    LAST_FOUR_PII,
    PII_FAMILY,
    PlanVersion,
    allowed_pii,
    label_violations,
)
from scripts.synth.scenarios import (
    INTENTS,
    MAX_REROLLS,
    legacy_scenarios,
    plan_report,
    reroll,
    scenarios,
)
from scripts.synth.schema import (
    Attempt,
    Draft,
    JudgeObservation,
    ObservedFlag,
    ObservedIntent,
    ScenarioV3,
)
from scripts.synth.storage import load_samples, read_jsonl

ROOT = Path(__file__).resolve().parents[2]


def test_tables_cover_exactly_the_77_intents() -> None:
    assert set(INTENT_KIND) == set(PII_FAMILY) == set(INTENTS)
    assert len(INTENTS) == 77
    assert Counter(INTENT_KIND.values()) == {"problem": 37, "info": 22, "request": 18}
    assert "CREDIT_CARD" not in allowed_pii("visa_or_mastercard")
    assert "IBAN_CODE" in allowed_pii("terminate_account")
    assert "IBAN_CODE" in allowed_pii("verify_source_of_funds")
    assert "CREDIT_CARD" not in allowed_pii("cancel_transfer")


@pytest.mark.parametrize("plan_version", [3, 4])
@pytest.mark.parametrize("seed", range(10))
def test_plans_keep_quotas_and_sample_only_plausible_targets(
    seed: int, plan_version: PlanVersion
) -> None:
    plan = scenarios(seed, plan=plan_version)
    assert plan == scenarios(seed, plan=plan_version) and len(plan) == 150
    assert all(
        isinstance(s, ScenarioV3)
        and s.plan_version == plan_version
        and s.id == f"synth-v{plan_version}-{seed}-{s.index:03d}"
        for s in plan
    )
    assert Counter(s.channel for s in plan) == {"chat": 100, "voice": 50}
    counts = Counter(s.intent for s in plan)
    assert set(counts) == set(INTENTS) and Counter(counts.values()) == {2: 73, 1: 4}
    assert {s.intent for s in plan if s.channel == "chat"} == set(INTENTS)
    assert Counter(s.resolved for s in plan) == {True: 75, False: 75}
    assert Counter(s.channel for s in plan if s.escalation_requested) == {"chat": 20, "voice": 10}
    assert sum(not s.slots for s in plan) == 30
    full = [slot for s in plan for slot in s.slots if slot.exposure == "full"]
    partial = [slot for s in plan for slot in s.slots if slot.exposure == "last_four"]
    assert len(full) == 240 and len(partial) == 30
    assert {slot.entity_type for slot in partial} <= LAST_FOUR_PII
    for s in plan:
        assert all(slot.entity_type in allowed_pii(s.intent) for slot in s.slots)
        assert not label_violations(
            s.intent,
            resolved=s.resolved,
            escalation_requested=s.escalation_requested,
            start=s.start_sentiment,
            end=s.end_sentiment,
            plan=plan_version,
        )
        if s.intent_kind == "info":
            assert not s.escalation_requested and s.start_sentiment != Sentiment.NEGATIVE
        if s.intent_kind == "problem":
            assert s.start_sentiment != Sentiment.POSITIVE
        if s.intent_kind == "info" and plan_version == 4:
            assert s.resolved
            assert (s.start_sentiment, s.end_sentiment) == (Sentiment.NEUTRAL, Sentiment.POSITIVE)
    unresolved_positive = [
        s
        for s in plan
        if not s.resolved and s.end_sentiment == "positive" and not s.escalation_requested
    ]
    assert len(unresolved_positive) == 3


def test_pii_types_stay_balanced_within_plausibility() -> None:
    report = plan_report(scenarios(42))
    assert sum(report.full_pii_by_type.values()) == 240
    assert set(report.full_pii_by_type) == {
        "PERSON",
        "CREDIT_CARD",
        "IBAN_CODE",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "ACCOUNT_NUMBER",
    }
    assert min(report.full_pii_by_type.values()) >= 20
    assert report.no_pii_controls == 30 and sum(report.last_four_by_type.values()) == 30
    assert report.intents_covered == 77


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"escalation_requested": True, "start_sentiment": "negative"}, "never escalate"),
        ({"start_sentiment": "negative"}, "cannot start negative"),
        (
            {
                "slots": [
                    {
                        "entity_type": "CREDIT_CARD",
                        "token": "[[CREDIT_CARD_1]]",
                        "turn_index": 0,
                        "speaker": "customer",
                    }
                ]
            },
            "not plausible",
        ),
        ({"intent_kind": "problem"}, "plausibility tables"),
    ],
)
def test_implausible_v3_scenarios_are_rejected(update: dict[str, object], message: str) -> None:
    info = next(s for s in scenarios(42) if s.intent == "visa_or_mastercard")
    with pytest.raises(ValidationError, match=message):
        ScenarioV3.model_validate({**info.model_dump(), **update})


def test_reroll_keeps_the_stratum_and_redraws_free_attributes() -> None:
    plan = scenarios(42)
    for base in plan[:30]:
        variants = [reroll(base, k) for k in range(MAX_REROLLS + 1)]
        assert variants[0] == base
        assert variants == [reroll(base, k) for k in range(MAX_REROLLS + 1)]
        for variant in variants[1:]:
            assert variant.id == base.id and variant.reroll > 0
            keep = (
                "intent",
                "channel",
                "resolved",
                "escalation_requested",
                "start_sentiment",
                "end_sentiment",
                "slots",
                "intent_kind",
                "pii_family",
            )
            assert {k: getattr(variant, k) for k in keep} == {k: getattr(base, k) for k in keep}
            assert variant.sample_seed != base.sample_seed
            assert [v.slot for v in values_for(variant)] == [v.slot for v in values_for(base)]
    free = ("customer_turns", "persona", "verbosity", "noise")
    assert any(
        {k: getattr(reroll(b, 1), k) for k in free} != {k: getattr(b, k) for k in free}
        for b in plan[:30]
    )
    with pytest.raises(ValueError):
        reroll(plan[0], MAX_REROLLS + 1)


def test_legacy_plan_reproduces_stored_v3_pilot_scenarios() -> None:
    stored = load_samples(ROOT / "data/synthetic/candidates/pilot-v3-haiku")
    legacy = legacy_scenarios(42)
    assert stored and all(s.labels.scenario == legacy[s.labels.scenario.index] for s in stored)


async def test_exhausted_slot_is_rerolled_then_left_unfilled(
    tmp_path: Path, settings: Settings, skeleton_draft: Callable
) -> None:
    base = scenarios(42)[0]

    def bad(k: int) -> dict[str, object]:
        draft = skeleton_draft(reroll(base, k)).model_dump()
        draft["turns"][0]["text"] = "Write to invented@example.com"
        return draft

    model = FakeStructuredChatModel(responses=[tool_answer(Draft, bad(k)) for k in range(3)])
    directory = tmp_path / "pilot"
    result = await generate_run(
        directory,
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=1,
        model_revision="test",
        max_attempts=1,
    )
    attempts = read_jsonl(directory / "attempts.jsonl", Attempt)
    assert [(a.attempt, a.reroll) for a in attempts] == [(1, 0), (2, 1), (3, 2)]
    assert result.unfilled_indices == [0] and result.rerolled_ids == [base.id]
    stats = result.rejections_by_stratum
    assert stats["intent_kind"][base.intent_kind].exhausted == 1
    for values in stats.values():
        assert sum(v.exhausted for v in values.values()) == len(result.unfilled_indices)
        assert sum(v.scenarios for v in values.values()) == 1
    assert stats["reroll"]["2"].attempts == 1
    assert stats["channel"]["chat"].reject_rate == 1.0
    again = await generate_run(
        directory,
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=1,
        model_revision="test",
        max_attempts=1,
        resume=True,
    )
    assert again.unfilled_indices == [0] and len(model.calls) == 3


async def test_reroll_admits_with_the_rerolled_scenario(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
) -> None:
    base = scenarios(42)[0]
    second = reroll(base, 1)
    bad = skeleton_draft(base).model_dump()
    bad["turns"][0]["text"] = "Write to invented@example.com"
    model = FakeStructuredChatModel(
        responses=[
            tool_answer(Draft, bad),
            tool_answer(Draft, skeleton_draft(second).model_dump()),
            tool_answer(JudgeObservation, judge_observation(second).model_dump()),
        ]
    )
    directory = tmp_path / "pilot"
    result = await generate_run(
        directory,
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=1,
        model_revision="test",
        max_attempts=1,
    )
    sample = load_samples(directory)[0]
    assert sample.labels.scenario == second and result.unfilled_indices == []
    assert result.admitted_free_attributes["reroll"] == {"1": 1}
    assert second.model_dump_json() in str(model.calls[1][1].content)


async def test_max_cost_stops_cleanly_and_resume_continues(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
) -> None:
    paid = settings.model_copy(
        update={"llm_provider": "anthropic", "llm_model": "claude-haiku-4-5"}
    )
    plan = scenarios(42)

    def answers(index: int) -> list:
        return [
            tool_answer(Draft, skeleton_draft(plan[index]).model_dump()),
            tool_answer(JudgeObservation, judge_observation(plan[index]).model_dump()),
        ]

    directory = tmp_path / "pilot"
    first = FakeStructuredChatModel(responses=answers(0))
    stopped = await generate_run(
        directory,
        LLMClient(paid, chat_model=first),
        paid,
        seed=42,
        n=2,
        model_revision="test",
        max_cost_usd=0.0001,
    )
    assert stopped.stopped_reason == "max_cost"
    assert stopped.candidates == 1 and stopped.missing_indices == [1]
    assert stopped.unfilled_indices == []
    second = FakeStructuredChatModel(responses=answers(1))
    resumed = await generate_run(
        directory,
        LLMClient(paid, chat_model=second),
        paid,
        seed=42,
        n=2,
        model_revision="test",
        resume=True,
    )
    assert resumed.stopped_reason is None and resumed.candidates == 2
    assert len(first.calls) == len(second.calls) == 2


async def test_max_cost_must_be_positive(tmp_path: Path, settings: Settings) -> None:
    client = LLMClient(settings, chat_model=FakeStructuredChatModel(responses=[]))
    with pytest.raises(ConvIntelError, match="positive"):
        await generate_run(
            tmp_path / "pilot",
            client,
            settings,
            seed=42,
            n=1,
            model_revision="test",
            max_cost_usd=0,
        )


def test_plan_4_moves_the_unresolved_quota_to_problem_and_request() -> None:
    plan = scenarios(42)
    unresolved = Counter(s.intent_kind for s in plan if not s.resolved)
    assert sum(unresolved.values()) == 75 and "info" not in unresolved
    assert {
        s.intent_kind
        for s in plan
        if not s.resolved and s.end_sentiment == "positive" and not s.escalation_requested
    } == {"request"}
    report = plan_report(plan)
    assert report.plan == "synth-v4" and report.start_by_kind["info"] == {"neutral": 42}


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"resolved": False}, "always resolved"),
        ({"end_sentiment": "neutral"}, "cannot end neutral"),
        ({"start_sentiment": "positive"}, "cannot start positive"),
    ],
)
def test_plan_4_rejects_the_informational_targets_that_failed_pilot_3_1(
    update: dict[str, object], message: str
) -> None:
    info = next(s for s in scenarios(42) if s.intent_kind == "info")
    with pytest.raises(ValidationError, match=message):
        ScenarioV3.model_validate({**info.model_dump(), **update})
    legacy = next(s for s in scenarios(42, plan=3) if s.intent_kind == "info")
    assert legacy.plan_version == 3


def _judged(
    scenario: ScenarioV3, labels: tuple[str, ...], observation: JudgeObservation
) -> JudgeObservation:
    flipped = {
        "resolved": ObservedFlag(
            value=not scenario.resolved, evidence_turns=observation.resolved.evidence_turns
        ),
        "intent": ObservedIntent(value="exchange_rate", evidence_turns=(0,)),
    }
    return observation.model_copy(update={label: flipped[label] for label in labels})


@pytest.mark.parametrize(
    ("failures", "systematic"),
    [
        ((("resolved",), ("resolved", "intent"), ("resolved",)), True),
        ((("resolved",), ("intent",), ("resolved",)), False),
    ],
)
async def test_same_field_base_failures_skip_rerolls(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
    failures: tuple[tuple[str, ...], ...],
    systematic: bool,
) -> None:
    base = scenarios(42)[0]
    assert base.intent != "exchange_rate"
    responses = []
    for labels in failures:
        responses += [
            tool_answer(Draft, skeleton_draft(base).model_dump()),
            tool_answer(
                JudgeObservation, _judged(base, labels, judge_observation(base)).model_dump()
            ),
        ]
    if not systematic:
        for k in (1, 1, 1, 2, 2, 2):
            variant = reroll(base, k)
            responses += [
                tool_answer(Draft, skeleton_draft(variant).model_dump()),
                tool_answer(
                    JudgeObservation,
                    _judged(variant, ("resolved",), judge_observation(variant)).model_dump(),
                ),
            ]
    model = FakeStructuredChatModel(responses=responses)
    directory = tmp_path / "pilot"
    result = await generate_run(
        directory,
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=1,
        model_revision="test",
        max_attempts=3,
    )
    attempts = read_jsonl(directory / "attempts.jsonl", Attempt)
    assert result.unfilled_indices == [0]
    if systematic:
        assert result.systematic_failures == {base.id: ["resolved"]}
        assert len(attempts) == 3 and result.rerolled_ids == []
    else:
        assert result.systematic_failures == {}
        assert len(attempts) == 9 and result.rerolled_ids == [base.id]
    assert not model.responses
