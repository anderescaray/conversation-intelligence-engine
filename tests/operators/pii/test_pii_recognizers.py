import pytest

from convintel.operators.pii import ACCOUNT_NUMBER, AccountNumberRecognizer
from convintel.operators.pii.recognizers import CONTEXT_SCORE, has_account_context


def _span(text: str, value: str) -> tuple[int, int]:
    start = text.index(value)
    return start, start + len(value)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("my account number is 83920174", "83920174"),
        ("ACCT: 0042817365", "0042817365"),
        ("the a/c 55102938 please", "55102938"),
        ("member id 77120093", "77120093"),
        ("acct no. 31950228", "31950228"),
        ("77310092 is the account I want closed", "77310092"),
    ],
)
def test_context_found(text: str, value: str) -> None:
    assert has_account_context(text, *_span(text, value))


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("order 58210394 arrived broken", "58210394"),
        ("account 99120458, call me on 312-555-0164", "312-555-0164"),
        ("call 312-555-0164 about account 99120458", "312-555-0164"),
        ("my account was charged, it was a long story and then order 58210394", "58210394"),
        ("call me at 415-555-0123. My account is overdue", "415-555-0123"),
    ],
)
def test_context_not_found(text: str, value: str) -> None:
    assert not has_account_context(text, *_span(text, value))


def test_scores_depend_on_context() -> None:
    recognizer = AccountNumberRecognizer()
    text = "account 83920174 and order 58210394"

    results = {
        text[r.start : r.end]: r.score for r in recognizer.analyze(text, [ACCOUNT_NUMBER], None)
    }

    assert results["83920174"] == CONTEXT_SCORE
    assert results["58210394"] < 0.1


def test_short_numbers_are_ignored() -> None:
    assert AccountNumberRecognizer().analyze("account 12345", [ACCOUNT_NUMBER], None) == []
