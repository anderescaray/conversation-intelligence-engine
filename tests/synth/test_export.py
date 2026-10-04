import random
import string
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from convintel.contracts import Conversation
from convintel.errors import ConvIntelError
from scripts.synth.review import export_golden
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import Annotation, ReviewEvent, Sample
from scripts.synth.storage import append_jsonl, digest, read_jsonl


def test_only_matching_human_approvals_export(
    tmp_path: Path,
    sample_factory: Callable,
    with_judge: Callable,
) -> None:
    samples: list[Sample] = []
    ledger = tmp_path / "review.jsonl"
    for scenario in scenarios(42):
        sample = sample_factory(scenario)
        rng = random.Random(scenario.sample_seed)
        unique = " ".join("".join(rng.choices(string.ascii_lowercase, k=10)) for _ in range(12))
        turns = sample.conversation.turns
        updated = turns[0].model_copy(update={"text": turns[0].text + " " + unique})
        sample = sample.model_copy(
            update={
                "conversation": sample.conversation.model_copy(
                    update={"turns": (updated, *turns[1:])}
                )
            }
        )
        sample = with_judge(sample)
        samples.append(sample)
        append_jsonl(
            ledger,
            ReviewEvent(
                id=sample.conversation.id,
                sample_sha256=digest(sample),
                reviewer="test-human",
                action="approve",
                at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        )
    output = tmp_path / "golden-v1"
    manifest = export_golden(samples, ledger, output)
    assert manifest.size == 150
    assert len(read_jsonl(output / "conversations.jsonl", Conversation)) == 150
    assert all(row.status == "approved" for row in read_jsonl(output / "labels.jsonl", Annotation))
    assert all(sample.labels.status == "candidate" for sample in samples)
    with pytest.raises(ConvIntelError, match="already exists"):
        export_golden(samples, ledger, output)
    append_jsonl(
        ledger,
        ReviewEvent(
            id=samples[0].conversation.id,
            sample_sha256=digest(samples[0]),
            reviewer="test-human",
            action="reject",
            note="Revoked after second review",
            at=datetime(2026, 1, 2, tzinfo=UTC),
        ),
    )
    with pytest.raises(ConvIntelError, match="150 human-approved"):
        export_golden(samples, ledger, tmp_path / "revoked")
