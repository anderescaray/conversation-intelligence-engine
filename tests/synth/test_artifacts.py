from pathlib import Path

from convintel.contracts import Conversation
from scripts.synth.audio import AudioManifest
from scripts.synth.pii import values_for
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
