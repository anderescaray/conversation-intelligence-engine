"""Shared test doubles for the pipeline tests."""

from datetime import UTC, datetime

from convintel.contracts import Channel, Conversation, Speaker, Turn

DEFAULT_TEXT = "Where is my new card?"


def make_conversation(
    *texts: str, conversation_id: str = "conv-1", redacted: bool = True
) -> Conversation:
    """A chat conversation with one customer turn per text.

    Redacted by default: the graph must only ever receive redacted conversations.
    """
    return Conversation(
        id=conversation_id,
        channel=Channel.CHAT,
        started_at=datetime(2026, 10, 20, tzinfo=UTC),
        turns=tuple(
            Turn(index=i, speaker=Speaker.CUSTOMER, text=text)
            for i, text in enumerate(texts or (DEFAULT_TEXT,))
        ),
        redacted=redacted,
    )
