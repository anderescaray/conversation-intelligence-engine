import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from convintel.errors import OperatorError
from convintel.operators.sentiment import (
    DEFAULT_MODEL_ID,
    PINNED_MODELS,
    HFSentimentModel,
    SentimentScores,
    model,
    provision_command,
)
from convintel.operators.sentiment.model import LabelMappingError

README = Path(model.__file__).with_name("README.md")


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


class FakeHub:
    """Stands in for `huggingface_hub` and `transformers`; records how the model is loaded."""

    def __init__(self, snapshot: Path, files: tuple[str, ...], error: OSError | None = None):
        self.snapshot = snapshot
        self.error = error
        self.snapshot_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.pipeline_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in files:
            (snapshot / name).write_text("{}")

    def snapshot_download(self, *args: Any, **kwargs: Any) -> str:
        self.snapshot_calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return str(self.snapshot)

    def pipeline(self, *args: Any, **kwargs: Any) -> FakePipeline:
        self.pipeline_calls.append((args, kwargs))
        return FakePipeline()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        modules = {
            "huggingface_hub": SimpleNamespace(snapshot_download=self.snapshot_download),
            "transformers": SimpleNamespace(pipeline=self.pipeline),
        }
        monkeypatch.setattr(model, "_import_module", modules.__getitem__)


PIN = PINNED_MODELS[DEFAULT_MODEL_ID]


def test_loads_the_pinned_revision_from_the_cache_by_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(tmp_path / "snapshot", PIN.files)
    hub.install(monkeypatch)

    loaded = model.load_hf_model(DEFAULT_MODEL_ID)

    assert hub.snapshot_calls == [
        ((DEFAULT_MODEL_ID,), {"revision": PIN.revision, "local_files_only": True})
    ]
    # A directory, not the repo id: a repo id lets transformers call the hub (auto-conversion).
    assert hub.pipeline_calls == [(("text-classification",), {"model": str(hub.snapshot)})]
    assert loaded.model_id == DEFAULT_MODEL_ID


def test_default_model_is_pinned_to_a_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", PIN.revision)
    assert "pytorch_model.bin" in PIN.files


def test_model_missing_from_the_cache_says_how_to_provision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # huggingface_hub's LocalEntryNotFoundError is a FileNotFoundError.
    hub = FakeHub(tmp_path / "snapshot", (), error=FileNotFoundError("not cached"))
    hub.install(monkeypatch)

    with pytest.raises(OperatorError) as excinfo:
        model.load_hf_model(DEFAULT_MODEL_ID)

    assert excinfo.value.operator == "sentiment"
    assert excinfo.value.message == (
        f"model {DEFAULT_MODEL_ID}@{PIN.revision} is missing or incomplete in the local "
        f"Hugging Face cache; provision it with: {provision_command(DEFAULT_MODEL_ID)}"
    )
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__
    assert hub.pipeline_calls == []


def test_incomplete_snapshot_fails_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(tmp_path / "snapshot", PIN.files[:-1])
    hub.install(monkeypatch)

    with pytest.raises(OperatorError, match="missing or incomplete"):
        model.load_hf_model(DEFAULT_MODEL_ID)

    assert hub.pipeline_calls == []


def test_unpinned_model_is_refused_without_touching_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(tmp_path / "snapshot", PIN.files)
    hub.install(monkeypatch)

    with pytest.raises(OperatorError, match="no pinned revision for model someone/other"):
        model.load_hf_model("someone/other")

    assert hub.snapshot_calls == []


def test_readme_documents_the_exact_provisioning_command() -> None:
    command = provision_command(DEFAULT_MODEL_ID)

    assert command.startswith(f"uv run --group ml hf download {DEFAULT_MODEL_ID} ")
    assert f"--revision {PIN.revision}" in command
    assert command in README.read_text()
