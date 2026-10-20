from pathlib import Path

import pytest

from convintel.contracts import Conversation
from scripts.synth.audio import AudioManifest
from scripts.synth.judge import reviewable
from scripts.synth.pii import values_for
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import Attempt, LegacyAnnotation, Report, RunSpec
from scripts.synth.storage import file_hash, load_samples, read_jsonl
from scripts.synth.validation import ensure_unique, validate_sample

ROOT = Path(__file__).resolve().parents[2]


def test_committed_candidates_and_audio_manifest_are_consistent() -> None:
    directory = ROOT / "data/synthetic/candidates/pilot-v1.1"
    samples = load_samples(directory)
    assert len(samples) == 10
    assert all(sample.labels.status == "candidate" for sample in samples)
    assert read_jsonl(directory / "conversations.jsonl", Conversation) == [
        sample.conversation for sample in samples
    ]
    assert read_jsonl(directory / "labels.jsonl", LegacyAnnotation) == [
        sample.labels for sample in samples
    ]
    for index, sample in enumerate(samples):
        validate_sample(sample)
        assert sample.labels.pii_values == values_for(sample.labels.scenario)
        ensure_unique(sample, samples[:index])
    manifest = AudioManifest.model_validate_json((directory / "audio-manifest.json").read_text())
    assert manifest.source_sha256 == file_hash(directory / "conversations.jsonl")
    assert manifest.channels == ("customer", "agent")
    assert (manifest.sample_rate, manifest.sample_width) == (16000, 2)
    assert manifest.customer_voice.model_id != manifest.agent_voice.model_id
    assert {row.conversation_id for row in manifest.files} == {
        sample.conversation.id for sample in samples if sample.conversation.channel == "voice"
    }
    for row in manifest.files:
        assert row.duration_s == row.frames / manifest.sample_rate
        assert len(row.sha256) == 64


def test_failed_v2_pilot_is_preserved_without_inventing_candidates() -> None:
    directory = ROOT / "data/synthetic/candidates/pilot-v2"
    run = RunSpec.model_validate_json((directory / "run.json").read_text())
    report = Report.model_validate_json((directory / "report.json").read_text())
    attempts = read_jsonl(directory / "attempts.jsonl", Attempt)
    assert run.generator_version == "2.0.0" and run.sample_timeout_s == 120
    assert report.target == 10 and report.candidates == report.human_approved == 0
    assert len(attempts) == report.attempts == report.rejects == 30
    assert report.rejection_categories == {"validator": 28, "judge": 2}
    assert all(not attempt.accepted for attempt in attempts)
    assert sum(attempt.judge is not None for attempt in attempts) == 2
    assert all(elapsed < 120 for elapsed in report.sample_elapsed_s.values())
    assert report.timed_out_ids == [] and report.missing_indices == list(range(10))
    assert load_samples(directory) == []
    assert read_jsonl(directory / "conversations.jsonl", Conversation) == []
    assert (directory / "labels.jsonl").read_text() == ""


@pytest.mark.parametrize(
    ("name", "generator", "admitted"),
    [("pilot-v3-haiku", "anthropic:claude-haiku-4-5", 8), ("pilot-v3-qwen", "ollama:qwen3:8b", 4)],
)
def test_v3_pilots_share_one_judge_and_publish_only_judge_passed_candidates(
    name: str, generator: str, admitted: int
) -> None:
    directory = ROOT / "data/synthetic/candidates" / name
    run = RunSpec.model_validate_json((directory / "run.json").read_text())
    report = Report.model_validate_json((directory / "report.json").read_text())
    samples = load_samples(directory)
    assert run.generator_version == "3.0.0" and run.seed == 42
    assert (report.model_id, report.judge_model_id) == (generator, "anthropic:claude-haiku-4-5")
    assert report.candidates == len(samples) == admitted and report.human_approved == 0
    assert read_jsonl(directory / "conversations.jsonl", Conversation) == [
        sample.conversation for sample in samples
    ]
    for index, sample in enumerate(samples):
        assert reviewable(sample)
        validate_sample(sample)
        ensure_unique(sample, samples[:index])
    attempts = read_jsonl(directory / "attempts.jsonl", Attempt)
    assert len(attempts) == report.attempts
    assert not {"pii_placement", "placeholder_inventory"} & set(report.rejection_reasons)


def test_v3_1_pilot_reports_unfilled_info_slots_without_relabeling() -> None:
    directory = ROOT / "data/synthetic/candidates/pilot-v3.1-haiku"
    run = RunSpec.model_validate_json((directory / "run.json").read_text())
    report = Report.model_validate_json((directory / "report.json").read_text())
    samples = load_samples(directory)
    plan = scenarios(42, plan=3)
    assert run.generator_version == "3.1.0" and run.max_rerolls == 2
    assert report.candidates == len(samples) == 7 and report.human_approved == 0
    assert report.unfilled_indices == [5, 7, 8] and report.stopped_reason is None
    assert report.rejections_by_stratum["intent_kind"]["info"].exhausted == 3
    for index, sample in enumerate(samples):
        assert sample.labels.scenario == plan[sample.labels.scenario.index]
        assert reviewable(sample)
        validate_sample(sample)
        ensure_unique(sample, samples[:index])


def test_v3_2_pilot_passed_the_full_run_gate() -> None:
    directory = ROOT / "data/synthetic/candidates/pilot-v3.2-haiku"
    run = RunSpec.model_validate_json((directory / "run.json").read_text())
    report = Report.model_validate_json((directory / "report.json").read_text())
    samples = load_samples(directory)
    plan = scenarios(42)
    assert run.generator_version == "3.2.0" and report.human_approved == 0
    assert report.candidates == len(samples) == 10
    assert report.unfilled_indices == [] and report.systematic_failures == {}
    assert all(
        stats.reject_rate < 1
        for values in report.rejections_by_stratum.values()
        for stats in values.values()
    )
    for index, sample in enumerate(samples):
        assert sample.labels.scenario == plan[sample.labels.scenario.index]
        assert reviewable(sample)
        validate_sample(sample)
        ensure_unique(sample, samples[:index])
