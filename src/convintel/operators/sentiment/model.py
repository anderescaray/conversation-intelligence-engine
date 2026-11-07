"""Sentiment model adapter and its process-wide lazy singleton.

The operator depends on the small `SentimentModel` protocol rather than on transformers, so
unit tests inject a fake and never import torch. `transformers` and torch are in the `ml`
dependency group (ADR-0005) and are imported only when the real model is loaded.

Nothing is downloaded from the Hugging Face hub at runtime (AGENTS.md rule 11): the model is
loaded from a pinned revision already in the local HF cache, provisioned once with
`provision_command()`. A missing or incomplete model fails warmup, and with it startup.
"""

import importlib
import threading
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from convintel.contracts import Sentiment
from convintel.errors import OperatorError

DEFAULT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
"""125M-parameter RoBERTa trained on short informal text, close to chat and ASR turns."""


class PinnedModel(BaseModel):
    """The exact commit of a hub model and the files that loading it needs."""

    model_config = ConfigDict(frozen=True)

    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: tuple[str, ...] = Field(min_length=1)


PINNED_MODELS: Mapping[str, PinnedModel] = {
    # Commit read from refs/main of the local HF cache on 2026-11-07, without a network call.
    # Weights are pytorch_model.bin: the repo's main branch has no safetensors file.
    DEFAULT_MODEL_ID: PinnedModel(
        revision="3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
        files=(
            "config.json",
            "merges.txt",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "vocab.json",
        ),
    ),
}
"""Models the operator may load. A model without a pinned revision is refused, never `main`."""

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


def provision_command(model_id: str) -> str:
    """The one-off command that puts the pinned revision of `model_id` in the HF cache."""
    pin = _pinned(model_id)
    return (
        f"uv run --group ml hf download {model_id} {' '.join(pin.files)} --revision {pin.revision}"
    )


def load_hf_model(model_id: str) -> HFSentimentModel:
    """Load the pinned revision of `model_id` from the local HF cache, on the default device.

    The pipeline gets the snapshot *directory*, not the repo id: given a repo id, transformers
    starts a safetensors auto-conversion thread that calls the hub even with
    `local_files_only=True`. A local directory never touches the hub.
    """
    pin = _pinned(model_id)
    missing = OperatorError(
        "sentiment",
        f"model {model_id}@{pin.revision} is missing or incomplete in the local Hugging Face "
        f"cache; provision it with: {provision_command(model_id)}",
    )
    hub = _import_module("huggingface_hub")
    try:
        # allow_patterns: a snapshot provisioned with only the pinned files is complete.
        # Without it, huggingface_hub checks the cache against the whole repo listing.
        snapshot = Path(
            hub.snapshot_download(
                model_id,
                revision=pin.revision,
                allow_patterns=list(pin.files),
                local_files_only=True,
            )
        )
    except OSError:
        # LocalEntryNotFoundError is an OSError; the fixed message says how to fix it.
        raise missing from None
    if not all((snapshot / name).is_file() for name in pin.files):
        raise missing
    transformers = _import_module("transformers")
    classifier = transformers.pipeline("text-classification", model=str(snapshot))
    return HFSentimentModel(classifier, model_id)


def _pinned(model_id: str) -> PinnedModel:
    try:
        return PINNED_MODELS[model_id]
    except KeyError:
        raise OperatorError("sentiment", f"no pinned revision for model {model_id}") from None


def _import_module(name: str) -> ModuleType:
    # Imported by name: transformers and huggingface_hub live in the `ml` group, so CI (and
    # pyright there) runs without them, and importing eagerly would pull torch into every
    # process.
    return importlib.import_module(name)


_lock = threading.Lock()


@cache
def _cached_model(model_id: str) -> SentimentModel:
    return load_hf_model(model_id)


def get_sentiment_model(model_id: str = DEFAULT_MODEL_ID) -> SentimentModel:
    """Shared model for `model_id`, loaded on first call; the lock prevents duplicate loads."""
    with _lock:
        return _cached_model(model_id)
