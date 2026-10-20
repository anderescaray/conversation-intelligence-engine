import argparse
import csv
from pathlib import Path

from pydantic import BaseModel

from convintel.contracts import Conversation

from .storage import file_hash, read_jsonl, write_json
from .taxonomy import DESCRIPTIONS
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


class DescriptionAudit(BaseModel):
    reference_sha256: str
    reference_rows: int
    taxonomy_version: int
    descriptions: int
    ngram: int
    containing_an_utterance: list[str]
    sharing_ngrams: dict[str, int]


def _grams(text: str, size: int) -> set[tuple[str, ...]]:
    words = text_key(text).split()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def description_overlaps(
    descriptions: dict[str, str], utterances: list[str], ngram: int = 6
) -> tuple[list[str], dict[str, int]]:
    """Intents whose description contains a whole utterance or shares an n-gram with one.

    Utterances shorter than four words are ignored for containment: they are generic fragments
    ("Where is my card?") that a paraphrase may legitimately share.
    """
    keys = [text_key(text) for text in utterances]
    reference = set().union(*(_grams(text, ngram) for text in utterances))
    containing = sorted(
        intent
        for intent, text in descriptions.items()
        if any(
            key and (key == text_key(text) or (len(key.split()) >= 4 and key in text_key(text)))
            for key in keys
        )
    )
    sharing = {
        intent: len(_grams(text, ngram) & reference)
        for intent, text in descriptions.items()
        if _grams(text, ngram) & reference
    }
    return containing, sharing


class Utterance(BaseModel):
    text: str


def read_reference(path: Path) -> list[str]:
    """Read utterances from a local CSV or JSONL snapshot with a `text` field; never downloads."""
    if path.suffix == ".jsonl":
        return [Utterance.model_validate_json(line).text for line in path.open(encoding="utf-8")]
    with path.open(encoding="utf-8") as stream:
        return [row["text"] for row in csv.DictReader(stream)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline holdout audit, never imported by generation"
    )
    parser.add_argument("--conversations", type=Path)
    parser.add_argument("--banking77-test-csv", type=Path)
    parser.add_argument(
        "--descriptions-against",
        type=Path,
        help="Audit intent descriptions against a local Banking77 CSV/JSONL snapshot instead",
    )
    parser.add_argument("--taxonomy-version", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Audit output already exists")
    if args.descriptions_against:
        reference = read_reference(args.descriptions_against)
        descriptions = DESCRIPTIONS[args.taxonomy_version]
        containing, sharing = description_overlaps(descriptions, reference)
        report = DescriptionAudit(
            reference_sha256=file_hash(args.descriptions_against),
            reference_rows=len(reference),
            taxonomy_version=args.taxonomy_version,
            descriptions=len(descriptions),
            ngram=6,
            containing_an_utterance=containing,
            sharing_ngrams=sharing,
        )
        write_json(args.output, report)
        print(report.model_dump_json(indent=2))
        return int(bool(containing or sharing))
    if args.conversations is None or args.banking77_test_csv is None:
        parser.error("--conversations and --banking77-test-csv are required")
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
