from pathlib import Path

from convintel.contracts import Conversation
from scripts.synth.audio import AudioManifest
from scripts.synth.pii import values_for
from scripts.synth.schema import Annotation
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
    assert read_jsonl(directory / "labels.jsonl", Annotation) == [
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
