"""Custom Presidio recognizers.

Account numbers have no checksum and look like any other long number (order IDs, amounts,
phone numbers without separators), so the regex alone is not evidence. A match only counts
when a context word sits next to it in the raw text.

Presidio's built-in context enhancer compares spaCy lemmas, which ties the result to the NLP
model (a blank or swapped pipeline has no lemmas). Checking a character window around the
match instead keeps the behaviour identical whatever spaCy model is loaded.
"""

import re

from presidio_analyzer import Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

ACCOUNT_NUMBER = "ACCOUNT_NUMBER"

_PATTERNS = [
    Pattern("account_digits", r"\b\d{6,17}\b", 0.05),
    Pattern("account_grouped", r"\b\d{2,6}(?:-\d{2,8}){1,3}\b", 0.05),
]

_CONTEXT = re.compile(
    r"\b(?:account|acct|a/c|member\s+(?:id|number|no)|customer\s+(?:id|number|no))",
    re.IGNORECASE,
)

_DIGIT = re.compile(r"\d")
_AFTER_STOP = re.compile(r"[\d.!?;]")
"""A keyword after the number must be in the same sentence ("1234 is my account")."""
_CLAIM_CHARS = 15
"""A keyword followed this closely by digits introduces that number, not the previous one."""

CONTEXT_WINDOW_CHARS = 40
"""Characters inspected on each side of a match: about six words of conversational text."""

CONTEXT_SCORE = 0.85
"""Score of a match with context. Above the redactor threshold, below checksum-validated hits."""


class AccountNumberRecognizer(PatternRecognizer):
    """Bank or customer account numbers: 6-17 digits, optionally hyphen-grouped, near a keyword."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity=ACCOUNT_NUMBER,
            patterns=_PATTERNS,
            name="AccountNumberRecognizer",
            supported_language="en",
        )

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
        regex_flags: int | None = None,
    ) -> list[RecognizerResult]:
        results = super().analyze(text, entities, nlp_artifacts, regex_flags)
        for result in results:
            if has_account_context(text, result.start, result.end):
                result.score = max(result.score, CONTEXT_SCORE)
        return results


def has_account_context(text: str, start: int, end: int) -> bool:
    """True when an account keyword is within the window around `text[start:end]`.

    The keyword must not be separated from the match by another number: in "account 1234,
    call 415-555-0123" the keyword belongs to 1234, not to the phone number. A keyword after
    the match must be in the same sentence, and not introduce a number of its own ("call
    415-555-0123 about account 1234"); one before it may not be ("acct no. 1234").
    """
    before = text[max(0, start - CONTEXT_WINDOW_CHARS) : start]
    keywords_before = list(_CONTEXT.finditer(before))
    if keywords_before and not _DIGIT.search(before[keywords_before[-1].end() :]):
        return True
    after = text[end : end + CONTEXT_WINDOW_CHARS]
    keyword_after = _CONTEXT.search(after)
    if keyword_after is None or _AFTER_STOP.search(after[: keyword_after.start()]):
        return False
    trailing = text[end + keyword_after.end() : end + keyword_after.end() + _CLAIM_CHARS]
    return not _DIGIT.search(trailing)
