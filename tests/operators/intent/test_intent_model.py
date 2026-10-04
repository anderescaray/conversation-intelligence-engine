from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from convintel.operators.intent import (
    META_FILE,
    IntentBundleMeta,
    IntentModelUnavailableError,
    get_intent_model,
    read_bundle_meta,
)
from convintel.operators.intent import model as model_module
from convintel.operators.intent.bundle import write_bundle_meta
from convintel.operators.intent.model import parse_registry_uri, resolve_model_uri

META = IntentBundleMeta(
    labels=("a", "b", "c"),
    temperature=1.7,
    threshold=0.8,
    max_length=16,
    base_model="tiny",
    dataset_version="test@0",
)


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("models:/intent-classifier@champion", ("intent-classifier", "champion", None)),
        ("models:/intent-classifier/7", ("intent-classifier", None, "7")),
    ],
)
def test_parse_registry_uri(uri: str, expected: tuple[str, str | None, str | None]) -> None:
    assert parse_registry_uri(uri) == expected


@pytest.mark.parametrize("uri", ["models:/", "models:/name", "models:/name/latest", "models:/@x"])
def test_parse_registry_uri_rejects_other_forms(uri: str) -> None:
    with pytest.raises(IntentModelUnavailableError, match="unsupported model URI"):
        parse_registry_uri(uri)


def test_bundle_meta_round_trips(tmp_path: Path) -> None:
    write_bundle_meta(tmp_path, META)

    assert read_bundle_meta(tmp_path) == META


def test_missing_bundle_says_how_to_get_one(tmp_path: Path) -> None:
    with pytest.raises(IntentModelUnavailableError, match=r"not found.*train"):
        read_bundle_meta(tmp_path / "nowhere")


def test_invalid_bundle_meta(tmp_path: Path) -> None:
    (tmp_path / META_FILE).write_text('{"labels": ["a"]}')

    with pytest.raises(IntentModelUnavailableError, match="invalid"):
        read_bundle_meta(tmp_path)


def test_a_local_path_is_used_as_is(tmp_path: Path) -> None:
    directory, model_id = resolve_model_uri(str(tmp_path / "bundle"), tmp_path, "unused")

    assert directory == tmp_path / "bundle"
    assert model_id == "local/bundle"


class FakeMlflow:
    """Just the MLflow surface `resolve_model_uri` uses; downloads write a fake bundle."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.downloads: list[str] = []
        self.artifacts = SimpleNamespace(download_artifacts=self._download)

    def MlflowClient(self, **kwargs: Any) -> Any:
        fake = self

        class Client:
            def get_model_version_by_alias(self, name: str, alias: str) -> Any:
                if fake.fail_with is not None:
                    raise fake.fail_with
                return SimpleNamespace(version=3)

            def get_model_version_download_uri(self, name: str, version: str) -> str:
                return f"mlflow-artifacts:/models/{name}/{version}"

        return Client()

    def _download(self, artifact_uri: str, dst_path: str, tracking_uri: str) -> str:
        self.downloads.append(artifact_uri)
        root = Path(dst_path) / "artifacts"
        bundle = root / "artifacts" / "bundle"
        bundle.mkdir(parents=True)
        write_bundle_meta(bundle, META)
        return str(root)


def _use_fake_mlflow(monkeypatch: pytest.MonkeyPatch, fake: FakeMlflow) -> None:
    real = model_module._import_ml  # pyright: ignore[reportPrivateUsage]

    def fake_import(name: str) -> ModuleType:
        return fake if name == "mlflow" else real(name)  # pyright: ignore[reportReturnType]

    monkeypatch.setattr(model_module, "_import_ml", fake_import)


def test_alias_is_resolved_and_downloaded_once_into_the_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeMlflow()
    _use_fake_mlflow(monkeypatch, fake)
    uri = "models:/intent-classifier@champion"

    first = resolve_model_uri(uri, tmp_path, "http://mlflow:5000")
    second = resolve_model_uri(uri, tmp_path, "http://mlflow:5000")

    (server,) = tmp_path.iterdir()
    expected_dir = server / "intent-classifier" / "v3" / "artifacts" / "bundle"
    assert first == second == (expected_dir, "intent-classifier/v3")
    assert read_bundle_meta(expected_dir) == META
    assert fake.downloads == ["mlflow-artifacts:/models/intent-classifier/3"]
    assert [p.name for p in (server / "intent-classifier").iterdir()] == ["v3"]


def test_the_same_version_on_another_server_is_a_different_cache_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeMlflow()
    _use_fake_mlflow(monkeypatch, fake)
    uri = "models:/intent-classifier@champion"

    first, _ = resolve_model_uri(uri, tmp_path, "http://mlflow:5000")
    second, _ = resolve_model_uri(uri, tmp_path, "sqlite:///other.db")

    assert first != second
    assert len(fake.downloads) == 2


def test_registry_errors_keep_only_the_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_fake_mlflow(monkeypatch, FakeMlflow(fail_with=ConnectionError("server said: <html>")))

    with pytest.raises(IntentModelUnavailableError) as info:
        resolve_model_uri("models:/intent-classifier@champion", tmp_path, "http://mlflow:5000")

    assert str(info.value) == (
        "cannot resolve models:/intent-classifier@champion from MLflow at "
        "http://mlflow:5000 (ConnectionError)"
    )
    assert info.value.__cause__ is None


def test_singleton_is_shared_and_failures_are_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Path] = []
    sentinel = object()

    def fake_load(directory: Path, model_id: str, device: str | None) -> object:
        calls.append(directory)
        if len(calls) == 1:
            raise IntentModelUnavailableError("not yet")
        return sentinel

    monkeypatch.setattr(model_module, "load_bundle", fake_load)

    with pytest.raises(IntentModelUnavailableError):
        get_intent_model(str(tmp_path), tmp_path)
    assert get_intent_model(str(tmp_path), tmp_path) is sentinel
    assert get_intent_model(str(tmp_path), tmp_path) is sentinel
    assert len(calls) == 2


def test_missing_ml_group_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> ModuleType:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(model_module, "importlib", SimpleNamespace(import_module=missing))

    with pytest.raises(IntentModelUnavailableError, match="uv sync --group ml"):
        model_module._import_ml("torch")  # pyright: ignore[reportPrivateUsage]
