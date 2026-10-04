"""Test doubles for the intent model. Texts are synthetic."""

import threading
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from convintel.contracts import Channel, Conversation, Speaker, Turn

LABELS: tuple[str, ...] = ("card_arrival", "lost_or_stolen_card", "exchange_rate", "top_up_failed")


class FakeIntentModel:
    """Returns canned logits keyed by text and records every batch it receives."""

    model_id: str = "fake/intent"
    labels: tuple[str, ...] = LABELS

    def __init__(
        self,
        logits: dict[str, Sequence[float]],
        temperature: float = 1.0,
        threshold: float | None = 0.9,
    ) -> None:
        self._logits = logits
        self.temperature = temperature
        self.threshold = threshold
        self.batches: list[list[str]] = []
        self.threads: list[int] = []

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.batches.append(list(texts))
        self.threads.append(threading.get_ident())
        return np.asarray([self._logits[t] for t in texts], dtype=np.float32)


class FailingIntentModel(FakeIntentModel):
    def __init__(self, exc: Exception) -> None:
        super().__init__({})
        self._exc = exc

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float32]:
        raise self._exc


def make_conversation(
    *turns: tuple[Speaker, str], conversation_id: str = "conv-intent"
) -> Conversation:
    return Conversation(
        id=conversation_id,
        channel=Channel.CHAT,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        turns=tuple(
            Turn(index=i, speaker=speaker, text=text) for i, (speaker, text) in enumerate(turns)
        ),
        redacted=True,
    )
