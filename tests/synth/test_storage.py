from collections.abc import Callable
from pathlib import Path

import pytest

from convintel.errors import ConvIntelError
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import Sample
from scripts.synth.storage import append_jsonl, load_samples, read_jsonl


def test_truncated_ledger_fails_without_exposing_content(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    path.write_text('{"conversation": "secret-synthetic-value"', encoding="utf-8")
    with pytest.raises(ConvIntelError, match="line 1") as error:
        read_jsonl(path, Sample)
    assert "secret-synthetic-value" not in str(error.value)


def test_latest_candidate_preserves_history(tmp_path: Path, sample_factory: Callable) -> None:
    sample = sample_factory(scenarios(42)[1])
    replacement = sample_factory(sample.labels.scenario.model_copy(update={"sample_seed": 111}))
    for record in (sample, replacement):
        append_jsonl(tmp_path / "candidates.jsonl", record)
    assert load_samples(tmp_path) == [replacement]
    assert len(read_jsonl(tmp_path / "candidates.jsonl", Sample)) == 2
