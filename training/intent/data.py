"""Banking77 on disk: pinned source, integrity manifest, loading and the validation split.

The data is fetched once by `download_data.py`; everything else only reads `DATA_DIR` and never
touches the network. The source is `mteb/banking77` at a fixed commit: the original
`PolyAI/banking77` repo only ships a loading script (no data files), which `datasets` no longer
runs, and the MTEB copy is the same train/test split as parquet.
"""

import hashlib
import importlib
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_REPO = "mteb/banking77"
DATASET_REVISION = "18072d2685ea682290f7b8924d94c62acc19c0b2"
Split = Literal["train", "test"]
SPLIT_FILES: dict[Split, str] = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
MANIFEST_FILE = "manifest.json"
NUM_CLASSES = 77


class DataUnavailableError(RuntimeError):
    """Banking77 is not on disk (or does not match the manifest): run download_data first."""


class FileEntry(BaseModel):
    path: str
    sha256: str
    size: int


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo: str
    revision: str
    files: dict[str, FileEntry]

    @property
    def version(self) -> str:
        """Tag value for MLflow `dataset.version`: source repo at its commit."""
        return f"{self.repo}@{self.revision}"


class LabeledTexts(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    texts: tuple[str, ...]
    labels: NDArray[np.int64]
    label_names: tuple[str, ...]

    def subset(self, indices: NDArray[np.int64]) -> "LabeledTexts":
        return LabeledTexts(
            texts=tuple(self.texts[i] for i in indices),
            labels=self.labels[indices],
            label_names=self.label_names,
        )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(data_dir: Path = DATA_DIR) -> DatasetManifest:
    path = data_dir / MANIFEST_FILE
    if not path.is_file():
        raise DataUnavailableError(
            f"{path} not found: run `uv run python -m training.intent.download_data` once"
        )
    return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_split(split: Split, data_dir: Path = DATA_DIR) -> LabeledTexts:
    """Read one official split from disk after checking its hash against the manifest."""
    manifest = read_manifest(data_dir)
    entry = manifest.files[split]
    path = data_dir / entry.path
    if not path.is_file() or sha256_of(path) != entry.sha256:
        raise DataUnavailableError(f"{path} is missing or modified: re-run download_data")
    datasets = importlib.import_module("datasets")
    table = datasets.Dataset.from_parquet(str(path))
    labels = np.asarray(table["label"], dtype=np.int64)
    return LabeledTexts(
        texts=tuple(table["text"]),
        labels=labels,
        label_names=label_names(labels, list(table["label_text"])),
    )


def label_names(labels: NDArray[np.int64], label_texts: list[str]) -> tuple[str, ...]:
    """Class names indexed by label id, checking that ids and names map one to one."""
    by_id: dict[int, str] = {}
    for label, text in zip(labels.tolist(), label_texts, strict=True):
        if by_id.setdefault(label, text) != text:
            raise ValueError(f"label {label} has two names: {by_id[label]!r} and {text!r}")
    if sorted(by_id) != list(range(NUM_CLASSES)):
        raise ValueError(f"expected label ids 0..{NUM_CLASSES - 1}, got {len(by_id)} ids")
    return tuple(by_id[i] for i in range(NUM_CLASSES))


def stratified_split(
    labels: NDArray[np.int64], val_fraction: float, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(train, validation) indices with each class split in the same proportion.

    Banking77 has no official validation split. Stratifying keeps all 77 classes in
    validation, which temperature fitting, ECE and τ selection need.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    val: list[int] = []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        rng.shuffle(members)
        val.extend(members[: max(1, round(len(members) * val_fraction))].tolist())
    val_idx = np.sort(np.asarray(val, dtype=np.int64))
    train_idx = np.setdiff1d(np.arange(len(labels), dtype=np.int64), val_idx)
    return train_idx, val_idx
