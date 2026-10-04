"""Pure parts of `training/intent`: split, config, data integrity and the calibration study."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from convintel.operators.intent import softmax
from training.intent import train
from training.intent.config import DEFAULT_CONFIG, TrainConfig, load_config
from training.intent.data import (
    MANIFEST_FILE,
    NUM_CLASSES,
    DatasetManifest,
    DataUnavailableError,
    FileEntry,
    label_names,
    load_split,
    read_manifest,
    stratified_split,
)
from training.intent.report import calibration_study, evaluate_frozen


def _labels(per_class: int = 20, n_classes: int = 7) -> np.ndarray:
    return np.repeat(np.arange(n_classes, dtype=np.int64), per_class)


def test_stratified_split_is_deterministic_disjoint_and_complete() -> None:
    labels = _labels()

    train_a, val_a = stratified_split(labels, 0.1, seed=42)
    train_b, val_b = stratified_split(labels, 0.1, seed=42)
    _, val_c = stratified_split(labels, 0.1, seed=7)

    np.testing.assert_array_equal(val_a, val_b)
    np.testing.assert_array_equal(train_a, train_b)
    assert not set(val_a) & set(train_a)
    assert sorted([*train_a, *val_a]) == list(range(len(labels)))
    assert not np.array_equal(val_a, val_c)


def test_every_class_keeps_its_share_in_validation() -> None:
    labels = np.concatenate([_labels(per_class=30, n_classes=5), np.array([5, 5])])

    _, val = stratified_split(labels, 0.1, seed=0)

    counts = np.bincount(labels[val], minlength=6)
    assert counts.tolist() == [3, 3, 3, 3, 3, 1]


def test_val_fraction_must_be_a_proper_fraction() -> None:
    with pytest.raises(ValueError):
        stratified_split(_labels(), 1.0, seed=0)


def test_label_names_follow_label_ids() -> None:
    labels = np.arange(NUM_CLASSES, dtype=np.int64)[::-1]
    names = [f"intent_{i}" for i in labels.tolist()]

    assert label_names(labels, names) == tuple(f"intent_{i}" for i in range(NUM_CLASSES))


def test_label_names_reject_conflicting_names() -> None:
    with pytest.raises(ValueError, match="two names"):
        label_names(np.array([0, 0]), ["a", "b"])


def test_default_config_is_valid_and_strict(tmp_path: Path) -> None:
    config = load_config()
    assert config.base_model == "distilbert-base-uncased"
    assert config.target_covered_accuracy == 0.97

    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(raw | {"learning_rte": 1e-4}))
    with pytest.raises(ValidationError):
        load_config(tmp_path / "c.yaml")
    assert set(raw) == set(TrainConfig.model_fields)


def test_missing_data_says_how_to_download(tmp_path: Path) -> None:
    with pytest.raises(DataUnavailableError, match="download_data"):
        read_manifest(tmp_path)


def test_modified_data_is_refused(tmp_path: Path) -> None:
    (tmp_path / "test.parquet").write_bytes(b"tampered")
    manifest = DatasetManifest(
        repo="r",
        revision="abc",
        files={"test": FileEntry(path="test.parquet", sha256="0" * 64, size=8)},
    )
    (tmp_path / MANIFEST_FILE).write_text(manifest.model_dump_json())

    with pytest.raises(DataUnavailableError, match="modified"):
        load_split("test", tmp_path)


def _overconfident(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 5, size=(n, 6))
    probs = softmax(logits, 3.0)
    labels = np.array([rng.choice(6, p=p) for p in probs], dtype=np.int64)
    return logits, labels


def test_calibration_study_fits_temperature_and_picks_tau() -> None:
    logits, labels = _overconfident(4000, seed=1)

    study = calibration_study(logits, labels, n_bins=15, target_accuracy=0.9, min_covered=50)

    assert study.temperature == pytest.approx(3.0, rel=0.15)
    assert study.ece_after < study.ece_before
    assert study.nll_after < study.nll_before
    assert study.threshold is not None and study.threshold.accuracy is not None
    assert study.threshold.accuracy >= 0.9
    assert 0 < study.threshold.coverage < 1


def test_frozen_evaluation_measures_without_choosing() -> None:
    val_logits, val_labels = _overconfident(4000, seed=1)
    test_logits, test_labels = _overconfident(2000, seed=2)
    study = calibration_study(
        val_logits, val_labels, n_bins=15, target_accuracy=0.9, min_covered=50
    )
    assert study.threshold is not None

    report = evaluate_frozen(
        test_logits,
        test_labels,
        temperature=study.temperature,
        threshold=study.threshold.threshold,
        n_bins=15,
        num_classes=6,
    )

    assert report.n == 2000
    assert report.ece_after < report.ece_before
    assert report.at_threshold is not None
    assert report.at_threshold.threshold == study.threshold.threshold


def _calls_load_split_test(node: ast.AST) -> bool:
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "load_split"
        and any(isinstance(a, ast.Constant) and a.value == "test" for a in call.args)
        for call in ast.walk(node)
    )


def test_only_final_evaluation_reads_the_test_split() -> None:
    tree = ast.parse(inspect.getsource(train))
    readers = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and _calls_load_split_test(node)
    ]

    assert readers == ["final_evaluation"]
    final = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "final_evaluation"
    )
    called = {
        c.func.id
        for c in ast.walk(final)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }
    assert "calibration_study" not in called
    assert "fit_temperature" not in called
