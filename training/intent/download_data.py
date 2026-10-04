"""One-time, explicit download of Banking77 to `training/intent/data/`.

    uv run --group ml python -m training.intent.download_data

The only step of T03 that fetches data. It pins the source commit and writes a manifest with
the sha256 of every file; training refuses to run on files that do not match it.
"""

import argparse
import importlib
from pathlib import Path

from .data import (
    DATA_DIR,
    DATASET_REPO,
    DATASET_REVISION,
    MANIFEST_FILE,
    SPLIT_FILES,
    DatasetManifest,
    FileEntry,
    load_split,
    sha256_of,
)


def download(data_dir: Path = DATA_DIR) -> DatasetManifest:
    hub = importlib.import_module("huggingface_hub")
    hub.snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=list(SPLIT_FILES.values()),
        local_dir=str(data_dir),
    )
    files: dict[str, FileEntry] = {}
    for split, relative in SPLIT_FILES.items():
        path = data_dir / relative
        files[split] = FileEntry(path=relative, sha256=sha256_of(path), size=path.stat().st_size)
    manifest = DatasetManifest(repo=DATASET_REPO, revision=DATASET_REVISION, files=files)
    (data_dir / MANIFEST_FILE).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Banking77 once to training/intent/data/."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    manifest = download(args.data_dir)
    for split in SPLIT_FILES:
        data = load_split(split, args.data_dir)
        print(f"{split}: {len(data.texts)} utterances, {len(set(data.labels.tolist()))} classes")
    print(f"dataset.version = {manifest.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
