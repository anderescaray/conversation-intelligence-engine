from collections.abc import Callable
from pathlib import Path

import pytest

from convintel.errors import ConvIntelError
from scripts.synth.review import decision, export_golden, review_batch, show_sample
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import ReviewEvent
from scripts.synth.storage import read_jsonl


def test_resumable_review_and_notes(tmp_path: Path, sample_factory: Callable) -> None:
    samples = [sample_factory(s) for s in scenarios(42)[:3]]
    ledger = tmp_path / "review.jsonl"
    answers = iter(["a", "", "n", "Check intent boundary"])
    assert (
        review_batch(
            samples, ledger, "human", 2, ask=lambda _: next(answers), display=lambda _: None
        )
        == 2
    )
    events = read_jsonl(ledger, ReviewEvent)
    approved = decision(samples[0], events)
    assert approved is not None and approved.action == "approve"
    assert decision(samples[1], events) is None
    shown: list[str] = []
    answers = iter(["r", "Incorrect intent", "q"])
    review_batch(samples, ledger, "human", ask=lambda _: next(answers), display=shown.append)
    assert samples[0].conversation.id not in " ".join(shown)
    assert "Check intent boundary" in " ".join(shown)
    rejected = decision(samples[1], read_jsonl(ledger, ReviewEvent))
    assert rejected is not None and rejected.action == "reject"
    assert all(sample.labels.status == "candidate" for sample in samples)


def test_changed_candidate_invalidates_approval(tmp_path: Path, sample_factory: Callable) -> None:
    sample = sample_factory(scenarios(42)[0])
    ledger = tmp_path / "review.jsonl"
    answers = iter(["a", ""])
    review_batch([sample], ledger, "human", ask=lambda _: next(answers), display=lambda _: None)
    events = read_jsonl(ledger, ReviewEvent)
    changed = sample.model_copy(
        update={
            "conversation": sample.conversation.model_copy(
                update={"metadata": {"synthetic": "true", "changed": "yes"}}
            )
        }
    )
    assert decision(sample, events) is not None
    assert decision(changed, events) is None
    assert "PII" in show_sample(sample_factory(scenarios(42)[1]))


def test_pending_and_rejected_samples_cannot_become_golden(
    tmp_path: Path, sample_factory: Callable
) -> None:
    samples = [sample_factory(scenarios(42)[0])]
    destination = tmp_path / "golden-v1"
    with pytest.raises(ConvIntelError, match="150 human-approved"):
        export_golden(samples, tmp_path / "review.jsonl", destination)
    assert not destination.exists()


def test_quit_and_invalid_input_do_not_approve(tmp_path: Path, sample_factory: Callable) -> None:
    ledger = tmp_path / "review.jsonl"
    answers = iter(["yes", "r", "", "q"])
    review_batch(
        [sample_factory(scenarios(42)[0])],
        ledger,
        "human",
        ask=lambda _: next(answers),
        display=lambda _: None,
    )
    assert read_jsonl(ledger, ReviewEvent) == []
