import re
import unicodedata
from collections.abc import Callable, Iterable

from convintel.errors import ConvIntelError

from .pii import TOKEN
from .scenarios import INTENTS
from .schema import Draft, Sample, Scenario

NameScanner = Callable[[str], bool]
_SUSPECT = re.compile(
    r"[\w.+-]+@[\w.-]+|\b\d[\d ()+-]{3,}\d\b|\b[A-Z]{2}\d{2}[A-Z0-9]{8,}\b|"
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine)(?:[ ,.-]+"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine)){3,}\b|"
    r"\b(?i:my name is|this is|i am|i'm)\s+[A-Z][a-z]+\s+[A-Z][a-z]+"
)
_ESCALATION = re.compile(r"\b(?:supervisor|manager|escalat\w*)\b", re.IGNORECASE)


class Rejected(ConvIntelError):
    pass


def text_key(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def check_unfilled(text: str) -> None:
    if re.search(r"[\[\]{}]|<[^>]+>", text):
        raise Rejected("leftover_placeholder")


def check_unowned_pii(text: str, scanner: NameScanner | None = None) -> None:
    if _SUSPECT.search(text) or (scanner is not None and scanner(text)):
        raise Rejected("unowned_pii")


def validate_draft(draft: Draft, scenario: Scenario, scanner: NameScanner | None = None) -> None:
    if scenario.intent not in INTENTS:
        raise Rejected("unknown_intent")
    if len(draft.turns) != 2 * scenario.customer_turns:
        raise Rejected("turn_count")
    expected = {slot.token for slot in scenario.slots}
    seen: list[str] = []
    for index, turn in enumerate(draft.turns):
        if turn.speaker != ("customer" if index % 2 == 0 else "agent"):
            raise Rejected("speaker_order")
        if not turn.text.strip():
            raise Rejected("empty_turn")
        seen.extend(TOKEN.findall(turn.text))
        safe = TOKEN.sub("redacted", turn.text)
        check_unfilled(safe)
        check_unowned_pii(safe, scanner)
    if set(seen) != expected or any(seen.count(token) > 3 for token in expected):
        raise Rejected("placeholder_inventory")
    requested = any(_ESCALATION.search(t.text) for t in draft.turns if t.speaker == "customer")
    if requested != scenario.escalation_requested:
        raise Rejected("escalation_evidence")


def masked_turns(sample: Sample) -> tuple[str, ...]:
    result = []
    for turn in sample.conversation.turns:
        text = turn.text
        spans = [s for s in sample.labels.pii_entities if s.turn_index == turn.index]
        for span in reversed(spans):
            text = text[: span.start] + "redacted" + text[span.end :]
        result.append(text)
    return tuple(result)


def validate_sample(sample: Sample, scanner: NameScanner | None = None) -> None:
    Sample.model_validate(sample.model_dump())
    for text in masked_turns(sample):
        check_unfilled(text)
        check_unowned_pii(text, scanner)


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
    return lambda text: any(entity.label_ == "PERSON" for entity in nlp(text).ents)
