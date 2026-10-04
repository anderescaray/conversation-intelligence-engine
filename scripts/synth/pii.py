import random
import re
from datetime import UTC, datetime, timedelta

from faker import Faker

from convintel.contracts import Channel, Conversation, Speaker, Turn

from .schema import Draft, LegacyScenario, PIISpan, PIIValue, Placement, Scenario

_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_SYMBOLS = {"@": "at", ".": "dot", "-": "dash", "_": "underscore", "+": "plus"}
TOKEN = re.compile(r"\[\[[A-Z_]+_[0-9]+\]\]")


def spoken(value: str, entity_type: str) -> str:
    if entity_type == "PERSON":
        return value
    return " ".join(
        _DIGITS[int(char)] if char.isdigit() else _SYMBOLS.get(char, char.upper())
        for char in value
        if char != " "
    )


def value_key(value: str, entity_type: str) -> str:
    normalized = value.casefold()
    if entity_type == "PERSON":
        return " ".join(re.findall(r"\w+", normalized))
    for word, count in (("double", 2), ("triple", 3)):
        normalized = re.sub(
            rf"\b{word}\s+({'|'.join(_DIGITS)}|oh|\d)\b",
            lambda match, count=count: " ".join([match[1]] * count),
            normalized,
        )
    for index, word in enumerate(_DIGITS):
        normalized = re.sub(rf"\b{word}\b", str(index), normalized)
    normalized = re.sub(r"\boh\b", "0", normalized)
    for symbol, word in _SYMBOLS.items():
        normalized = re.sub(rf"\b{word}\b", symbol, normalized)
    if entity_type in {"PHONE_NUMBER", "CREDIT_CARD", "ACCOUNT_NUMBER"}:
        return re.sub(r"\D", "", normalized)
    return re.sub(
        r"[^a-z0-9@._-]" if entity_type == "EMAIL_ADDRESS" else r"[^a-z0-9]", "", normalized
    )


def values_for(scenario: Scenario | LegacyScenario) -> tuple[PIIValue, ...]:
    rng = random.Random(scenario.sample_seed)
    fake = Faker("en_US")
    fake.seed_instance(scenario.sample_seed)
    factories = {
        "PERSON": lambda: f"{fake.first_name()} {fake.last_name()}",
        "CREDIT_CARD": lambda: rng.choice(("4242424242424242", "5555555555554444")),
        "IBAN_CODE": lambda: rng.choice(("GB82WEST12345698765432", "DE89370400440532013000")),
        "PHONE_NUMBER": lambda: f"+1 202 555 {rng.randint(100, 199):04d}",
        "EMAIL_ADDRESS": lambda: f"user{rng.randint(100, 999)}@example.com",
        "ACCOUNT_NUMBER": lambda: f"9900{rng.randrange(1000000):06d}",
    }
    values: dict[str, PIIValue] = {}
    ordered = sorted(
        scenario.slots, key=lambda s: isinstance(s, Placement) and s.exposure == "last_four"
    )
    for slot in ordered:
        canonical = (
            values[slot.source_token].canonical_value[-4:]
            if isinstance(slot, Placement) and slot.source_token is not None
            else factories[slot.entity_type]()
        )
        surface = spoken(canonical, slot.entity_type)
        values[slot.token] = PIIValue(
            slot=slot,
            canonical_value=canonical,
            spoken_form=surface,
            surface_form=surface if scenario.channel == "voice" else canonical,
        )
    return tuple(values[slot.token] for slot in scenario.slots)


def insert_pii(
    draft: Draft,
    scenario: Scenario,
) -> tuple[Conversation, tuple[PIIValue, ...], tuple[PIISpan, ...]]:
    values = values_for(scenario)
    by_token = {value.slot.token: value for value in values}
    turns: list[Turn] = []
    spans: list[PIISpan] = []
    for index, turn in enumerate(draft.turns):
        parts: list[str] = []
        cursor = offset = 0
        for match in TOKEN.finditer(turn.text):
            prefix = turn.text[cursor : match.start()]
            parts.append(prefix)
            offset += len(prefix)
            value = by_token[match[0]]
            parts.append(value.surface_form)
            end = offset + len(value.surface_form)
            spans.append(
                PIISpan(
                    entity_type=value.slot.entity_type,
                    turn_index=index,
                    start=offset,
                    end=end,
                    slot_token=match[0],
                )
            )
            offset, cursor = end, match.end()
        parts.append(turn.text[cursor:])
        turns.append(Turn(index=index, speaker=Speaker(turn.speaker), text="".join(parts)))
    conversation = Conversation(
        id=scenario.id,
        channel=Channel(scenario.channel),
        language="en",
        started_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=scenario.index),
        turns=tuple(turns),
        redacted=False,
        metadata={
            "synthetic": "true",
            "seed": str(scenario.seed),
            "sample_seed": str(scenario.sample_seed),
        },
    )
    return conversation, values, tuple(spans)
