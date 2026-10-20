import re
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel

from convintel.config import Settings
from convintel.errors import ConvIntelError
from convintel.llm import LLMClient, load_prompt
from convintel.llm.testing import FakeStructuredChatModel, tool_answer
from scripts.synth.audit import description_overlaps
from scripts.synth.generate import generate_run
from scripts.synth.judge import JUDGE_TAXONOMY, reviewable
from scripts.synth.merge import merge
from scripts.synth.rejudge import RejudgeRecord, rejudge
from scripts.synth.scenarios import legacy_scenarios, scenarios
from scripts.synth.schema import (
    Annotation,
    Attempt,
    Draft,
    JudgeObservation,
    ObservedFlag,
    RunSpec,
    ScenarioV3,
)
from scripts.synth.storage import digest, load_samples, read_jsonl, write_json, write_jsonl
from scripts.synth.taxonomy import (
    CONFUSABLE_FAMILIES,
    DESCRIPTIONS,
    INTENT_DESCRIPTIONS_V1,
    INTENT_DESCRIPTIONS_V2,
    INTENTS,
)
from scripts.synth.validation import redact_unowned

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "evals/datasets/labeling-guide-v1.md"


class Utterance(BaseModel):
    text: str


def test_generator_and_judge_embed_the_same_guide_verbatim() -> None:
    guide = GUIDE.read_text()
    assert guide in load_prompt("synthetic", 4).system
    assert guide in load_prompt("synthetic_judge", 3).system
    for rule in (
        "An escalation is resolved only if the person it is escalated to resolves it",
        "Courtesy is neutral",
        "Calmly reporting a bad event is neutral",
    ):
        assert rule in guide


def test_guide_descriptions_are_the_code_taxonomy() -> None:
    listed = dict(
        re.findall(r"^- (\S+): (.+)$", GUIDE.read_text().split("## Confusable intents")[1], re.M)
    )
    assert len(listed) == 15 and set(CONFUSABLE_FAMILIES) <= set(listed)
    assert all(INTENT_DESCRIPTIONS_V2[name] == text for name, text in listed.items())
    assert "PIN" in listed["get_physical_card"]


def test_taxonomy_v2_changes_only_audited_descriptions_and_is_bound_to_judge_v3() -> None:
    changed = {
        name for name in INTENTS if INTENT_DESCRIPTIONS_V1[name] != INTENT_DESCRIPTIONS_V2[name]
    }
    assert len(changed) == 21
    assert {"get_physical_card", "apple_pay_or_google_pay", "visa_or_mastercard"} <= changed
    assert set(DESCRIPTIONS[1]) == set(DESCRIPTIONS[2]) == set(INTENTS)
    assert JUDGE_TAXONOMY == {2: 1, 3: 2}
    plan_v1, plan_v2 = scenarios(42, taxonomy=1), scenarios(42)
    for old, new in zip(plan_v1, plan_v2, strict=True):
        assert old.id == new.id and old.taxonomy_version == 1 and new.taxonomy_version == 2
        keep = old.model_dump(exclude={"intent_description", "taxonomy_version"})
        assert keep == new.model_dump(exclude={"intent_description", "taxonomy_version"})
    assert all(
        s.intent_description == INTENT_DESCRIPTIONS_V1[s.intent] for s in legacy_scenarios(42)
    )


def test_description_leakage_audit() -> None:
    reference = [
        Utterance.model_validate_json(line).text
        for line in (ROOT / "tests/synth/banking77_test_fixture.jsonl").open()
    ]
    for version in (1, 2):
        assert description_overlaps(DESCRIPTIONS[version], reference) == ([], {})
    copied = {"card_arrival": f"Customer asks: {reference[0]}"}
    containing, _ = description_overlaps(copied, reference)
    assert containing == ["card_arrival"]


def test_rejected_text_keeps_markers_and_redacts_invented_pii() -> None:
    text = "My card is [[CREDIT_CARD_1]] and write to me at me@mail.test or 4111 1111 1111 1111."
    redacted = redact_unowned(text)
    assert "[[CREDIT_CARD_1]]" in redacted and "me@mail.test" not in redacted
    assert "4111" not in redacted and redacted.count("[[UNOWNED_PII]]") == 2
    assert (
        redact_unowned("I am [[PERSON_1]], hi Jane", lambda t: "Jane" in t) == "[[REDACTED_TURN]]"
    )


async def test_judge_rejection_stores_masked_text_without_faker_values(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
) -> None:
    scenario = scenarios(42)[1]
    observation = judge_observation(scenario)
    observation = observation.model_copy(
        update={"resolved": ObservedFlag(value=not scenario.resolved, evidence_turns=(3,))}
    )
    model = FakeStructuredChatModel(
        responses=[
            tool_answer(Draft, skeleton_draft(scenario).model_dump()),
            tool_answer(JudgeObservation, observation.model_dump()),
        ]
    )
    directory = tmp_path / "pilot"
    await generate_run(
        directory,
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=2,
        model_revision="test",
        max_attempts=1,
        max_rerolls=0,
        indices=[1],
    )
    attempt = read_jsonl(directory / "attempts.jsonl", Attempt)[0]
    assert attempt.reason == "label_mismatch" and attempt.masked_turns is not None
    stored = " ".join(turn.text for turn in attempt.masked_turns)
    assert all(slot.token in stored for slot in scenario.slots)
    raw = (directory / "attempts.jsonl").read_text()
    from scripts.synth.pii import values_for

    for value in values_for(scenario):
        assert value.canonical_value not in raw


async def test_indices_and_dedup_limit_a_regeneration(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
    sample_factory: Callable,
) -> None:
    scenario = scenarios(42)[3]
    model = FakeStructuredChatModel(
        responses=[
            tool_answer(Draft, skeleton_draft(scenario).model_dump()),
            tool_answer(JudgeObservation, judge_observation(scenario).model_dump()),
        ]
    )
    directory = tmp_path / "regen"
    result = await generate_run(
        directory,
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=150,
        model_revision="test",
        indices=[3],
    )
    run = RunSpec.model_validate_json((directory / "run.json").read_text())
    assert run.indices == [3] and result.missing_indices == [] and result.candidates == 1
    duplicate = FakeStructuredChatModel(
        responses=[tool_answer(Draft, skeleton_draft(scenario).model_dump())]
    )
    deduped = await generate_run(
        tmp_path / "dedup",
        LLMClient(settings, chat_model=duplicate),
        settings,
        seed=42,
        n=150,
        model_revision="test",
        indices=[3],
        max_attempts=1,
        max_rerolls=0,
        dedup_against=load_samples(directory),
    )
    assert deduped.rejection_reasons == {"duplicate_dialogue": 1}
    with pytest.raises(ConvIntelError, match="subset"):
        await generate_run(
            tmp_path / "bad",
            LLMClient(settings),
            settings,
            seed=42,
            n=2,
            model_revision="t",
            indices=[5],
        )


def _source(tmp_path: Path, samples: list) -> Path:
    directory = tmp_path / "source"
    directory.mkdir()
    write_json(
        directory / "run.json",
        RunSpec(
            seed=42,
            model_id="anthropic:claude-haiku-4-5",
            model_revision="test",
            prompt_sha256="a" * 64,
            judge_model_id="anthropic:claude-haiku-4-5",
            judge_model_revision="test",
            judge_prompt_sha256="c" * 64,
            environment_sha256="b" * 64,
            pii_scan="patterns",
        ),
    )
    write_jsonl(directory / "candidates.jsonl", samples)
    return directory


async def test_rejudge_keeps_text_and_labels_and_reports_flips_by_field(
    tmp_path: Path,
    settings: Settings,
    sample_factory: Callable,
    judge_observation: Callable,
) -> None:
    plan = scenarios(42, taxonomy=1)
    kept, flipped = sample_factory(plan[2]), sample_factory(plan[3])
    source = _source(tmp_path, [kept, flipped])
    wrong = judge_observation(plan[3]).model_copy(
        update={"resolved": ObservedFlag(value=not plan[3].resolved, evidence_turns=(3,))}
    )
    model = FakeStructuredChatModel(
        responses=[
            tool_answer(JudgeObservation, judge_observation(plan[2]).model_dump()),
            tool_answer(JudgeObservation, wrong.model_dump()),
        ]
    )
    output = tmp_path / "rejudged"
    report = await rejudge(source, output, LLMClient(settings, chat_model=model))
    assert (report.kept, report.flipped, report.flipped_indices) == (1, 1, [3])
    assert report.flips_by_field == {"resolved": 1} and report.taxonomy_version == 2
    sample = load_samples(output)[0]
    assert sample.conversation == kept.conversation and reviewable(sample)
    assert isinstance(sample.labels, Annotation) and isinstance(sample.labels.scenario, ScenarioV3)
    assert sample.labels.scenario.taxonomy_version == 2
    assert sample.labels.scenario.model_dump(
        exclude={"intent_description", "taxonomy_version"}
    ) == (kept.labels.scenario.model_dump(exclude={"intent_description", "taxonomy_version"}))
    assert sample.labels.judge is not None and sample.labels.judge.prompt_version.endswith("v3")
    records = read_jsonl(output / "rejudge.jsonl", RejudgeRecord)
    assert [r.reason for r in records] == ["kept", "label_mismatch"]
    with pytest.raises(ConvIntelError, match="Output exists"):
        await rejudge(source, output, LLMClient(settings, chat_model=model))


async def test_rejudge_stops_cleanly_at_the_cost_cap(
    tmp_path: Path, settings: Settings, sample_factory: Callable, judge_observation: Callable
) -> None:
    plan = scenarios(42, taxonomy=1)
    source = _source(tmp_path, [sample_factory(plan[2]), sample_factory(plan[3])])
    paid = settings.model_copy(
        update={"llm_provider": "anthropic", "llm_model": "claude-haiku-4-5"}
    )
    model = FakeStructuredChatModel(
        responses=[tool_answer(JudgeObservation, judge_observation(plan[2]).model_dump())]
    )
    report = await rejudge(
        source, tmp_path / "out", LLMClient(paid, chat_model=model), max_cost_usd=0.0001
    )
    assert report.stopped_reason == "max_cost" and report.judged == 1


def test_merge_combines_sources_and_reports_coverage(
    tmp_path: Path, sample_factory: Callable
) -> None:
    plan = scenarios(42)
    first, second = tmp_path / "a", tmp_path / "b"
    for directory, samples in (
        (first, [sample_factory(plan[0])]),
        (second, [sample_factory(plan[1])]),
    ):
        directory.mkdir()
        write_jsonl(directory / "candidates.jsonl", samples)
    report = merge([first, second], tmp_path / "merged")
    assert report.size == 2 and report.unfilled_indices == list(range(2, 150))
    assert len(report.intents_missing) == 77 - len({plan[0].intent, plan[1].intent})
    assert digest(load_samples(tmp_path / "merged")[0]) == digest(load_samples(first)[0])
    with pytest.raises(ConvIntelError, match="more than one source"):
        merge([first, first], tmp_path / "again")
