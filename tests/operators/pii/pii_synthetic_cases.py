"""Labelled synthetic PII test set. Every value is fictitious.

Phones use the 555-01xx range reserved for fiction, emails use the reserved example.* domains,
cards are published issuer test numbers and IBANs are the registry's documentation examples.
Names are invented and also listed in `SYNTHETIC_NAMES`, which the unit tests feed to a
deny-list recognizer standing in for the spaCy NER model.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from convintel.contracts import Channel, Conversation, PIIEntity, Speaker, Turn


@dataclass(frozen=True)
class PIICase:
    id: str
    text: str
    expected: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    """(entity_type, exact substring of `text`) pairs that must be redacted."""
    must_keep: list[str] = field(default_factory=list[str])
    """Substrings that are not PII and must survive redaction."""


SYNTHETIC_NAMES = [
    "Marisol Quintavalle",
    "Tobias Wrenfield",
    "Ingrid Halvorsen-Pike",
    "Dmitri Okonkwo",
    "Priya Thistlewood",
    "Lucas Fernhill",
    "Amara Delacroix-Nwosu",
]

CASES: list[PIICase] = [
    PIICase(
        "phone_us_dashes",
        "You can reach me at 415-555-0123 after five.",
        [("PHONE_NUMBER", "415-555-0123")],
    ),
    PIICase(
        "phone_us_parens",
        "My cell is (212) 555-0187, call anytime.",
        [("PHONE_NUMBER", "(212) 555-0187")],
    ),
    PIICase(
        "phone_e164",
        "Please text +1 617 555 0142 when it ships.",
        [("PHONE_NUMBER", "+1 617 555 0142")],
    ),
    PIICase(
        "phone_uk",
        "The landline is 020 7946 0958 if the mobile fails.",
        [("PHONE_NUMBER", "020 7946 0958")],
    ),
    PIICase(
        "email_plain",
        "Send the receipt to marisol.q@example.com please.",
        [("EMAIL_ADDRESS", "marisol.q@example.com")],
    ),
    PIICase(
        "email_plus_tag",
        "I used tobias+billing@example.org when I signed up.",
        [("EMAIL_ADDRESS", "tobias+billing@example.org")],
    ),
    PIICase(
        "card_visa_spaces",
        "Charge it to 4111 1111 1111 1111 like last time.",
        [("CREDIT_CARD", "4111 1111 1111 1111")],
    ),
    PIICase(
        "card_mastercard_compact",
        "The card number is 5555555555554444 and it expires soon.",
        [("CREDIT_CARD", "5555555555554444")],
    ),
    PIICase(
        "card_amex_dashes",
        "Use my amex 3782-822463-10005 for the deposit.",
        [("CREDIT_CARD", "3782-822463-10005")],
    ),
    PIICase(
        "iban_gb_spaced",
        "Refund to GB82 WEST 1234 5698 7654 32 please.",
        [("IBAN_CODE", "GB82 WEST 1234 5698 7654 32")],
    ),
    PIICase(
        "iban_de_compact",
        "My IBAN is DE89370400440532013000.",
        [("IBAN_CODE", "DE89370400440532013000")],
    ),
    PIICase(
        "account_number_keyword",
        "My account number is 83920174 and the payment bounced.",
        [("ACCOUNT_NUMBER", "83920174")],
    ),
    PIICase(
        "account_acct_abbrev",
        "Acct 0042817365 shows a double charge.",
        [("ACCOUNT_NUMBER", "0042817365")],
    ),
    PIICase(
        "account_grouped",
        "It's customer number 4417-002931 on the letter.",
        [("ACCOUNT_NUMBER", "4417-002931")],
    ),
    PIICase(
        "account_after_number",
        "The number 77310092 is the account I want closed.",
        [("ACCOUNT_NUMBER", "77310092")],
    ),
    PIICase(
        "person_full_name",
        "Hi, this is Marisol Quintavalle calling about my order.",
        [("PERSON", "Marisol Quintavalle")],
    ),
    PIICase(
        "person_hyphenated",
        "The order is under Ingrid Halvorsen-Pike.",
        [("PERSON", "Ingrid Halvorsen-Pike")],
    ),
    PIICase(
        "person_and_email",
        "Tobias Wrenfield here, my email is t.wrenfield@example.net.",
        [("PERSON", "Tobias Wrenfield"), ("EMAIL_ADDRESS", "t.wrenfield@example.net")],
    ),
    PIICase(
        "person_phone_account",
        "I'm Dmitri Okonkwo, account 99120458, call me on 312-555-0164.",
        [
            ("PERSON", "Dmitri Okonkwo"),
            ("ACCOUNT_NUMBER", "99120458"),
            ("PHONE_NUMBER", "312-555-0164"),
        ],
    ),
    PIICase(
        "card_and_iban",
        "Not the card 4012 8888 8888 1881, refund to GB33BUKB20201555555555 instead.",
        [("CREDIT_CARD", "4012 8888 8888 1881"), ("IBAN_CODE", "GB33BUKB20201555555555")],
    ),
    PIICase(
        "spoken_style",
        "yeah so my name's Priya Thistlewood and my number is 415 555 0199",
        [("PERSON", "Priya Thistlewood"), ("PHONE_NUMBER", "415 555 0199")],
    ),
    PIICase(
        "two_people",
        "Lucas Fernhill set it up but Amara Delacroix-Nwosu pays the bill.",
        [("PERSON", "Lucas Fernhill"), ("PERSON", "Amara Delacroix-Nwosu")],
    ),
    PIICase(
        "order_number_not_account",
        "Order 58210394 arrived broken, the box was crushed.",
        must_keep=["58210394"],
    ),
    PIICase(
        "no_pii",
        "I'd like to upgrade to the premium plan for 12 months.",
        must_keep=["premium plan", "12 months"],
    ),
]


def make_conversation(*texts: str, conversation_id: str = "conv-test") -> Conversation:
    """Alternating customer/agent turns, starting with the customer."""
    speakers = (Speaker.CUSTOMER, Speaker.AGENT)
    return Conversation(
        id=conversation_id,
        channel=Channel.CHAT,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        turns=tuple(
            Turn(index=i, speaker=speakers[i % 2], text=text) for i, text in enumerate(texts)
        ),
    )


def recalled(case: PIICase, entities: list[PIIEntity], redacted_text: str) -> list[bool]:
    """Per expected value: True if it was redacted and labelled with the expected type.

    A value counts as found only if a detected span of that type covers it entirely and the
    value no longer appears in the redacted text.
    """
    hits: list[bool] = []
    for entity_type, value in case.expected:
        start = case.text.index(value)
        end = start + len(value)
        covered = any(
            e.entity_type == entity_type and e.start <= start and e.end >= end for e in entities
        )
        hits.append(covered and value not in redacted_text)
    return hits
