import threading
from typing import Any

import pytest

from convintel.operators.sentiment import HFSentimentModel, SentimentScores, model
from convintel.operators.sentiment.model import LabelMappingError


class FakePipeline:
    """Mimics `pipeline("text-classification")(texts, top_k=None, ...)` output."""

    def __init__(self, labels: tuple[str, str, str] = ("negative", "neutral", "positive")):
        self.labels = labels
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append((texts, kwargs))
        return [
            [
                {"label": self.labels[2], "score": 0.7},
                {"label": self.labels[1], "score": 0.2},
                {"label": self.labels[0], "score": 0.1},
            ]
            for _ in texts
        ]


def test_one_batched_call_with_all_class_scores() -> None:
    pipeline = FakePipeline()

    scores = HFSentimentModel(pipeline, "m", batch_size=8).predict(["a", "b", "c"])

    assert scores == [SentimentScores(negative=0.1, neutral=0.2, positive=0.7)] * 3
    assert len(pipeline.calls) == 1
    texts, kwargs = pipeline.calls[0]
    assert texts == ["a", "b", "c"]
    assert kwargs == {"top_k": None, "truncation": True, "batch_size": 8}


def test_empty_input_skips_the_model() -> None:
    pipeline = FakePipeline()

    assert HFSentimentModel(pipeline, "m").predict([]) == []
    assert pipeline.calls == []


def test_labels_are_case_insensitive() -> None:
    scores = HFSentimentModel(FakePipeline(("Negative", "Neutral", "Positive")), "m").predict(["x"])

    assert scores[0].positive == 0.7


def test_unknown_labels_raise() -> None:
    with pytest.raises(LabelMappingError, match="label_0"):
        HFSentimentModel(FakePipeline(("LABEL_0", "LABEL_1", "LABEL_2")), "m").predict(["x"])


def test_get_sentiment_model_loads_once_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[str] = []
    loaded = HFSentimentModel(FakePipeline(), "m")

    def fake_load(model_id: str) -> HFSentimentModel:
        loads.append(model_id)
        return loaded

    monkeypatch.setattr(model, "load_hf_model", fake_load)
    model._cached_model.cache_clear()  # pyright: ignore[reportPrivateUsage]
    try:
        got: list[object] = []
        workers = [
            threading.Thread(target=lambda: got.append(model.get_sentiment_model("m")))
            for _ in range(8)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    finally:
        model._cached_model.cache_clear()  # pyright: ignore[reportPrivateUsage]

    assert loads == ["m"]
    assert all(g is loaded for g in got)
