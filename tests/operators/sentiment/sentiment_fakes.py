"""Test doubles for the sentiment model. Texts are synthetic."""

import threading
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock

from convintel.contracts import Channel, Conversation, Speaker, Turn
from convintel.operators.sentiment import SentimentScores

NEG = SentimentScores(negative=0.9, neutral=0.08, positive=0.02)
NEU = SentimentScores(negative=0.1, neutral=0.8, positive=0.1)
POS = SentimentScores(negative=0.02, neutral=0.08, positive=0.9)


class FakeSentimentModel:
    """Returns canned scores keyed by text and records every batch it receives."""

    model_id = "fake/sentiment"

    def __init__(self, scores: dict[str, SentimentScores]) -> None:
        self._scores = scores
        self.batches: list[list[str]] = []
        self.threads: list[int] = []

    def predict(self, texts: Sequence[str]) -> list[SentimentScores]:
        self.batches.append(list(texts))
        self.threads.append(threading.get_ident())
        return [self._scores[t] for t in texts]


class RecordingTracer:
    """Records exceptions the way the OTel SDK does: the full traceback, causes included."""

    def __init__(self) -> None:
        self.recorded: list[str] = []

    @contextmanager
    def start_as_current_span(self, name: str) -> Iterator[MagicMock]:
        try:
            yield MagicMock()
        except BaseException as exc:
            self.recorded.append("".join(traceback.format_exception(exc)))
            raise


def make_conversation(*turns: tuple[Speaker, str]) -> Conversation:
    return Conversation(
        id="conv-sentiment",
        channel=Channel.CHAT,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        turns=tuple(
            Turn(index=i, speaker=speaker, text=text) for i, (speaker, text) in enumerate(turns)
        ),
        redacted=True,
    )
