import re
import threading
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

from convintel.errors import ConvIntelError

from .pii import TOKEN, value_key, values_for
from .schema import (
    Draft,
    JudgeRecord,
    Placement,
    Sample,
    Scenario,
    ScenarioV3,
    SkeletonDraft,
    SkeletonTurn,
)
from .taxonomy import DESCRIPTIONS, TaxonomyVersion

NameScanner = Callable[[str], bool]
_SUSPECT = re.compile(
    r"[\w.+-]+@[\w.-]+|\b\d[\d ()+-]{2,}\d\b|\b[A-Z]{2}\d{2}[A-Z0-9]{8,}\b|"
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine)(?:[ ,.-]+"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine)){3,}\b|"
    r"\b(?i:my name is|this is|i am|i'm)\s+[A-Z][a-z]+\s+[A-Z][a-z]+"
)
_ESCALATION = re.compile(r"\b(?:supervisor|manager|escalat\w*)\b", re.IGNORECASE)


class Rejected(ConvIntelError):
    def __init__(
        self,
        reason: str,
        *,
        category: Literal["validator", "judge"] = "validator",
        judge: JudgeRecord | None = None,
    ) -> None:
        super().__init__(reason)
        self.category: Literal["validator", "judge"] = category
        self.judge = judge


def text_key(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def check_unfilled(text: str) -> None:
    if re.search(r"[\[\]{}]|<[^>]+>", text):
        raise Rejected("leftover_placeholder")
    if re.search(r"[\x00-\x08\x0b-\x1f\x7f]", text):
        raise Rejected("control_characters")


def redact_unowned(text: str, scanner: NameScanner | None = None) -> str:
    """Keep a rejected turn reviewable without persisting personal data the model invented.

    Code-owned placeholders stay as markers; anything resembling PII outside them is replaced,
    and a turn in which the name scanner still finds a person is withheld entirely.
    """
    parts = TOKEN.split(text)
    tokens = TOKEN.findall(text)
    redacted = [_SUSPECT.sub("[[UNOWNED_PII]]", part) for part in parts]
    joined = "".join(
        segment + (tokens[index] if index < len(tokens) else "")
        for index, segment in enumerate(redacted)
    )
    if scanner is not None and scanner(
        TOKEN.sub("redacted", joined.replace("[[UNOWNED_PII]]", ""))
    ):
        return "[[REDACTED_TURN]]"
    return joined


def check_unowned_pii(text: str, scanner: NameScanner | None = None) -> None:
    if _SUSPECT.search(text) or (scanner is not None and scanner(text)):
        raise Rejected("unowned_pii")


def reject_repetition(texts: Iterable[str]) -> None:
    keys = tuple(text_key(text) for text in texts)
    for width in range(2, len(keys) // 2 + 1):
        for start in range(len(keys) - 2 * width + 1):
            block = keys[start : start + width]
            if any(
                keys[other : other + width] == block
                for other in range(start + width, len(keys) - width + 1)
            ):
                raise Rejected("repeated_block")
    if len(set(keys)) != len(keys):
        raise Rejected("repeated_turn")


def skeleton_errors(draft: SkeletonDraft, plan: Sequence[SkeletonTurn]) -> list[str]:
    """Concrete mismatches against the code-built skeleton, phrased for a single repair call."""
    errors: list[str] = []
    if len(draft.turns) != len(plan):
        errors.append(f"expected exactly {len(plan)} turns, got {len(draft.turns)}")
    for index, (turn, expected) in enumerate(zip(draft.turns, plan, strict=False)):
        if turn.turn_index != expected.turn_index:
            errors.append(f"entry {index} must have turn_index {expected.turn_index}")
        if turn.speaker != expected.speaker:
            errors.append(f"turn {index} must be spoken by the {expected.speaker}")
        found = Counter(TOKEN.findall(turn.text))
        for token in expected.placeholders:
            if found[token] != 1:
                errors.append(
                    f"turn {index} ({expected.speaker}) must contain {token} exactly once; "
                    f"found {found[token]}"
                )
        for token in sorted(set(found) - set(expected.placeholders)):
            errors.append(f"turn {index} ({expected.speaker}) must not contain {token}")
    return errors


def validate_draft(draft: Draft, scenario: Scenario, scanner: NameScanner | None = None) -> None:
    taxonomy: TaxonomyVersion = scenario.taxonomy_version if isinstance(scenario, ScenarioV3) else 1
    if DESCRIPTIONS[taxonomy].get(scenario.intent) != scenario.intent_description:
        raise Rejected("intent_description")
    if not 6 <= len(draft.turns) <= 10 or len(draft.turns) != 2 * scenario.customer_turns:
        raise Rejected("turn_count")
    seen: list[tuple[str, int, str]] = []
    safe_turns: list[str] = []
    for index, turn in enumerate(draft.turns):
        if turn.speaker != ("customer" if index % 2 == 0 else "agent"):
            raise Rejected("speaker_order")
        if not turn.text.strip():
            raise Rejected("empty_turn")
        if scenario.intent.casefold() in turn.text.casefold():
            raise Rejected("intent_label_leak")
        seen.extend((token, index, turn.speaker) for token in TOKEN.findall(turn.text))
        safe = TOKEN.sub("redacted", turn.text)
        check_unfilled(safe)
        check_unowned_pii(safe, scanner)
        safe_turns.append(safe)
    expected = [(slot.token, slot.turn_index, slot.speaker) for slot in scenario.slots]
    if Counter(seen) != Counter(expected):
        reason = (
            "pii_placement"
            if Counter(t[0] for t in seen) == Counter(t[0] for t in expected)
            else "placeholder_inventory"
        )
        raise Rejected(reason)
    reject_repetition(safe_turns)
    requested = any(_ESCALATION.search(t.text) for t in draft.turns if t.speaker == "customer")
    if requested != scenario.escalation_requested:
        raise Rejected("escalation_evidence")


def masked_turns(sample: Sample, *, typed: bool = False) -> tuple[str, ...]:
    result = []
    for turn in sample.conversation.turns:
        text = turn.text
        spans = [s for s in sample.labels.pii_entities if s.turn_index == turn.index]
        for span in reversed(spans):
            text = (
                text[: span.start] + (span.slot_token if typed else "redacted") + text[span.end :]
            )
        result.append(text)
    return tuple(result)


def validate_sample(sample: Sample, scanner: NameScanner | None = None) -> None:
    Sample.model_validate(sample.model_dump())
    texts = masked_turns(sample)
    for text in texts:
        check_unfilled(text)
        check_unowned_pii(text, scanner)
    if isinstance(sample.labels.scenario, Scenario):
        if sample.labels.pii_values != values_for(sample.labels.scenario):
            raise Rejected("pii_factory_mismatch")
        reject_repetition(texts)
        for value in sample.labels.pii_values:
            if isinstance(value.slot, Placement) and value.slot.exposure == "full":
                key = value_key(value.canonical_value, value.slot.entity_type)
                if any(
                    key in value_key(t.text, value.slot.entity_type)
                    for t in sample.conversation.turns
                    if t.speaker == "agent"
                ):
                    raise Rejected("agent_full_pii")


def ensure_unique(sample: Sample, previous: Iterable[Sample]) -> None:
    current = text_key(" ".join(masked_turns(sample)))
    words = current.split()
    grams = {tuple(words[i : i + 5]) for i in range(max(0, len(words) - 4))}
    for other in previous:
        prior = text_key(" ".join(masked_turns(other)))
        tokens = prior.split()
        other_grams = {tuple(tokens[i : i + 5]) for i in range(max(0, len(tokens) - 4))}
        union = grams | other_grams
        if current == prior or (union and len(grams & other_grams) / len(union) >= 0.85):
            raise Rejected("duplicate_dialogue")


def local_name_scanner(model: str) -> NameScanner:
    import spacy

    if not spacy.util.is_package(model):
        raise ConvIntelError(
            "NER model is not installed; provision it explicitly with the ml group"
        )
    nlp = spacy.load(model, disable=["parser", "lemmatizer"])
    lock = threading.Lock()

    def contains_name(text: str) -> bool:
        with lock:
            return any(entity.label_ == "PERSON" for entity in nlp(text).ents)

    return contains_name
