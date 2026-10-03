"""Sentiment model adapter and its process-wide lazy singleton.

The operator depends on the small `SentimentModel` protocol rather than on transformers, so
unit tests inject a fake and never import torch. `transformers` is imported only when the real
model is loaded; it needs torch at that point (the `ml` group locally, see ADR-0005).
"""

import threading
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any, Protocol

from pydantic import BaseModel, Field

from convintel.contracts import Sentiment

DEFAULT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
"""125M-parameter RoBERTa trained on short informal text, close to chat and ASR turns."""

DEFAULT_BATCH_SIZE = 16

TextClassifier = Callable[..., Any]
"""A transformers text-classification pipeline, or anything called the same way."""


class SentimentScores(BaseModel):
    """Class probabilities for one turn."""

    negative: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    positive: float = Field(ge=0, le=1)

    @property
    def label(self) -> Sentiment:
        probabilities = {
            Sentiment.NEGATIVE: self.negative,
            Sentiment.NEUTRAL: self.neutral,
            Sentiment.POSITIVE: self.positive,
        }
        return max(probabilities, key=probabilities.__getitem__)

    @property
    def polarity(self) -> float:
        """Expected sentiment in [-1, 1]: confident negatives weigh more than hesitant ones."""
        return self.positive - self.negative


class SentimentModel(Protocol):
    model_id: str

    def predict(self, texts: Sequence[str]) -> list[SentimentScores]:
        """Scores for each text, in order. Blocking: call it from a worker thread."""
        ...


class LabelMappingError(ValueError):
    """The model's labels are not negative/neutral/positive."""


class HFSentimentModel:
    """Wraps a transformers pipeline; one batched call per conversation."""

    def __init__(
        self, classifier: TextClassifier, model_id: str, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        self._classifier = classifier
        self.model_id = model_id
        self._batch_size = batch_size

    def predict(self, texts: Sequence[str]) -> list[SentimentScores]:
        if not texts:
            return []
        raw = self._classifier(
            list(texts), top_k=None, truncation=True, batch_size=self._batch_size
        )
        return [_to_scores(labels) for labels in raw]


def _to_scores(labels: list[dict[str, Any]]) -> SentimentScores:
    by_label = {str(item["label"]).lower(): float(item["score"]) for item in labels}
    try:
        return SentimentScores(
            negative=by_label["negative"],
            neutral=by_label["neutral"],
            positive=by_label["positive"],
        )
    except KeyError as exc:
        raise LabelMappingError(f"unexpected sentiment labels {sorted(by_label)}") from exc


def load_hf_model(model_id: str) -> HFSentimentModel:
    """Download (first time) and load `model_id` on CPU."""
    from transformers import pipeline

    return HFSentimentModel(pipeline("text-classification", model=model_id), model_id)


_lock = threading.Lock()


@cache
def _cached_model(model_id: str) -> SentimentModel:
    return load_hf_model(model_id)


def get_sentiment_model(model_id: str = DEFAULT_MODEL_ID) -> SentimentModel:
    """Shared model for `model_id`, loaded on first call; the lock prevents duplicate loads."""
    with _lock:
        return _cached_model(model_id)
