import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ValidationError

from convintel.errors import ConvIntelError

from .schema import Sample


def digest(record: BaseModel) -> str:
    return hashlib.sha256(
        json.dumps(record.model_dump(mode="json"), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl[T: BaseModel](path: Path, schema: type[T]) -> list[T]:
    if not path.exists():
        return []
    records: list[T] = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            try:
                records.append(schema.model_validate_json(line))
            except ValidationError:
                raise ConvIntelError(f"Invalid JSONL record at line {index}") from None
    return records


def append_jsonl(path: Path, record: BaseModel) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(record.model_dump_json() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(record.model_dump_json() + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(path: Path, record: BaseModel) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_samples(directory: Path) -> list[Sample]:
    samples = read_jsonl(directory / "candidates.jsonl", Sample)
    latest = {sample.conversation.id: sample for sample in samples}
    return sorted(latest.values(), key=lambda sample: sample.labels.scenario.index)
