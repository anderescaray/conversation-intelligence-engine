import argparse
import csv
from pathlib import Path

from pydantic import BaseModel

from convintel.contracts import Conversation

from .storage import file_hash, read_jsonl, write_json
from .validation import text_key


class Match(BaseModel):
    conversation_id: str
    turn_index: int


class Audit(BaseModel):
    reference_sha256: str
    reference_rows: int
    conversation_count: int
    matches: list[Match]


def exact_test_matches(conversations: list[Conversation], utterances: list[str]) -> list[Match]:
    reference = {text_key(text) for text in utterances}
    return [
        Match(conversation_id=c.id, turn_index=t.index)
        for c in conversations
        for t in c.turns
        if t.speaker == "customer" and text_key(t.text) in reference
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline holdout audit, never imported by generation"
    )
    parser.add_argument("--conversations", type=Path, required=True)
    parser.add_argument("--banking77-test-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Audit output already exists")
    with args.banking77_test_csv.open(encoding="utf-8") as stream:
        utterances = [row["text"] for row in csv.DictReader(stream)]
    conversations = read_jsonl(args.conversations, Conversation)
    if not utterances or not conversations:
        parser.error("Audit inputs must not be empty")
    result = Audit(
        reference_sha256=file_hash(args.banking77_test_csv),
        reference_rows=len(utterances),
        conversation_count=len(conversations),
        matches=exact_test_matches(conversations, utterances),
    )
    write_json(args.output, result)
    print(result.model_dump_json(indent=2))
    return int(bool(result.matches))


if __name__ == "__main__":
    raise SystemExit(main())
