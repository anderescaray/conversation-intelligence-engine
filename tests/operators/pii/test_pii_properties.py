"""Property-based checks: no generated PII value survives redaction, whatever the mix."""

import asyncio

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pii_synthetic_cases import SYNTHETIC_NAMES, make_conversation
from presidio_analyzer import AnalyzerEngine

from convintel.operators.pii import PIIRedactor

_SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _luhn_complete(prefix: str) -> str:
    """Append the check digit that makes `prefix` pass the Luhn test."""
    total = 0
    for i, ch in enumerate(reversed(prefix)):
        d = int(ch)
        if i % 2 == 0:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return prefix + str((10 - total % 10) % 10)


def _iban_gb(bank: str, digits: str) -> str:
    bban = bank + digits
    rearranged = "".join(str(int(c, 36)) for c in bban + "GB00")
    return f"GB{98 - int(rearranged) % 97:02d}{bban}"


def _group(value: str, sep: str) -> str:
    return sep.join(value[i : i + 4] for i in range(0, len(value), 4))


_digits = st.text(alphabet="0123456789", min_size=1, max_size=1)

phones = st.builds(
    lambda area, line, fmt: fmt.format(a=area, n=line),
    st.sampled_from(["212", "312", "415", "617", "646", "718"]),
    st.integers(100, 199).map(lambda n: f"0{n}"),
    st.sampled_from(["{a}-555-{n}", "({a}) 555-{n}", "+1 {a} 555 {n}", "{a}.555.{n}"]),
)
emails = st.builds(
    lambda local, domain: f"{local}@{domain}",
    st.from_regex(r"[a-z]{2,10}(\.[a-z]{2,8})?", fullmatch=True),
    st.sampled_from(["example.com", "example.org", "example.net"]),
)
cards = st.builds(
    lambda body, sep: _group(_luhn_complete("4" + body), sep),
    st.text(alphabet="0123456789", min_size=14, max_size=14),
    st.sampled_from(["", " ", "-"]),
)
ibans = st.builds(
    lambda bank, digits, sep: _group(_iban_gb(bank, digits), sep),
    st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=4, max_size=4),
    st.text(alphabet="0123456789", min_size=14, max_size=14),
    st.sampled_from(["", " "]),
)
accounts = st.builds(
    lambda first, rest: first + rest,
    st.sampled_from("123456789"),
    st.text(alphabet="0123456789", min_size=7, max_size=11),
)
names = st.sampled_from(SYNTHETIC_NAMES)

pii_snippets = st.one_of(
    phones.map(lambda v: (v, f"you can call me at {v}")),
    emails.map(lambda v: (v, f"my email is {v}")),
    cards.map(lambda v: (v, f"charge the card {v}")),
    ibans.map(lambda v: (v, f"refund it to {v}")),
    accounts.map(lambda v: (v, f"my account number is {v}")),
    names.map(lambda v: (v, f"this is {v} speaking")),
)
fillers = st.sampled_from(
    ["", "hello there", "sorry about the wait", "thanks so much", "ok great", "um yes"]
)


@st.composite
def turns_with_pii(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    """1-4 turns, each mixing 0-3 PII snippets with filler; returns (turn texts, PII values)."""
    texts: list[str] = []
    values: list[str] = []
    for _ in range(draw(st.integers(1, 4))):
        parts = [draw(fillers)]
        for value, sentence in draw(st.lists(pii_snippets, max_size=3)):
            values.append(value)
            parts.append(sentence)
            parts.append(draw(fillers))
        texts.append(". ".join(p for p in parts if p))
    return texts, values


@_SETTINGS
@given(turns_with_pii())
def test_no_original_pii_survives(
    analyzer: AnalyzerEngine, data: tuple[list[str], list[str]]
) -> None:
    texts, values = data

    result = asyncio.run(PIIRedactor(analyzer=analyzer).run(make_conversation(*texts)))

    redacted = result.output.redacted_conversation
    for value in values:
        for turn in redacted.turns:
            assert value not in turn.text, f"leaked {value!r} in {turn.text!r}"


@_SETTINGS
@given(st.lists(st.text(max_size=200), min_size=1, max_size=5))
def test_structure_preserved_for_arbitrary_text(analyzer: AnalyzerEngine, texts: list[str]) -> None:
    conversation = make_conversation(*texts)

    result = asyncio.run(PIIRedactor(analyzer=analyzer).run(conversation))

    redacted = result.output.redacted_conversation
    assert redacted.redacted is True
    assert [(t.index, t.speaker) for t in redacted.turns] == [
        (t.index, t.speaker) for t in conversation.turns
    ]
    for entity in result.output.entities:
        original = conversation.turns[entity.turn_index].text
        assert 0 <= entity.start < entity.end <= len(original)
