from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from convintel.contracts import (
    Channel,
    Conversation,
    OperatorResult,
    Sentiment,
    SentimentOutput,
    Speaker,
    Turn,
)


def _conversation() -> Conversation:
    # Synthetic content only: fixtures must never contain real PII.
    return Conversation(
        id="conv-test-001",
        channel=Channel.CHAT,
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        turns=(
            Turn(index=0, speaker=Speaker.CUSTOMER, text="My card was declined."),
            Turn(index=1, speaker=Speaker.AGENT, text="Let me check that for you."),
        ),
    )


def test_conversation_text_joins_turns_with_speakers() -> None:
    assert _conversation().text == (
        "customer: My card was declined.\nagent: Let me check that for you."
    )


def test_conversation_is_immutable() -> None:
    conversation = _conversation()
    with pytest.raises(ValidationError):
        conversation.id = "other"  # type: ignore[misc]


def test_turn_rejects_negative_index() -> None:
    with pytest.raises(ValidationError):
        Turn(index=-1, speaker=Speaker.AGENT, text="hi")


def test_operator_result_wraps_typed_output() -> None:
    result = OperatorResult[SentimentOutput](
        operator="sentiment",
        operator_version="0.1.0",
        output=SentimentOutput(overall=Sentiment.NEGATIVE),
        latency_ms=12.5,
    )
    assert result.output.overall is Sentiment.NEGATIVE
    assert result.cost_usd is None
