"""Intent model adapter, model resolution (local path or MLflow registry) and the singleton.

The operator depends on the small `IntentModel` protocol rather than on transformers, so unit
tests inject a fake and never import torch. torch, transformers and mlflow are in the `ml`
dependency group (ADR-0005) and are imported by name only when a real model is loaded.

Nothing is downloaded from the Hugging Face hub at runtime: the tokenizer and weights come
from the bundle (`local_files_only=True`). A `models:/` URI is resolved against the project's
own MLflow server, once per model version, into a local cache.
"""

import hashlib
import importlib
import shutil
import tempfile
import threading
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from convintel.config import get_settings

from .bundle import META_FILE, IntentBundleMeta, IntentModelUnavailableError, read_bundle_meta

REGISTERED_MODEL_NAME = "intent-classifier"
CHAMPION_ALIAS = "champion"
DEFAULT_MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
BUNDLE_ARTIFACT = "bundle"
"""Key of the bundle in the registered pyfunc model's `artifacts/` directory."""

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "convintel" / "models"
DEFAULT_BATCH_SIZE = 32


class IntentModel(Protocol):
    model_id: str
    labels: tuple[str, ...]
    temperature: float
    threshold: float | None

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Raw (uncalibrated) logits, shape (len(texts), len(labels)). Blocking."""
        ...


class HFIntentModel:
    """A fine-tuned sequence classifier and its tokenizer, batched under `inference_mode`."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        meta: IntentBundleMeta,
        model_id: str,
        device: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_length = meta.max_length
        self._device = device
        self._batch_size = batch_size
        self.model_id = model_id
        self.labels = meta.labels
        self.temperature = meta.temperature
        self.threshold = meta.threshold

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float32]:
        torch = _import_ml("torch")
        if not texts:
            return np.empty((0, len(self.labels)), dtype=np.float32)
        batches: list[NDArray[np.float32]] = []
        for start in range(0, len(texts), self._batch_size):
            encoded = self._tokenizer(
                list(texts[start : start + self._batch_size]),
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                logits = self._model(**encoded).logits
            batches.append(logits.float().cpu().numpy())
        return np.concatenate(batches)


def _import_ml(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        raise IntentModelUnavailableError(
            f"the intent model needs '{name}': run `uv sync --group ml`"
        ) from None


def load_bundle(directory: Path, model_id: str, device: str | None = None) -> HFIntentModel:
    """Load a bundle from disk on `device` (default: CUDA when torch sees it, else CPU)."""
    meta = read_bundle_meta(directory)
    torch = _import_ml("torch")
    transformers = _import_ml("transformers")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        directory, local_files_only=True
    )
    if model.config.num_labels != len(meta.labels):
        raise IntentModelUnavailableError(
            f"bundle at {directory} has {model.config.num_labels} logits "
            f"but {len(meta.labels)} labels"
        )
    model.to(device).eval()
    return HFIntentModel(model, tokenizer, meta, model_id=model_id, device=device)


def parse_registry_uri(uri: str) -> tuple[str, str | None, str | None]:
    """`models:/name@alias` → (name, alias, None); `models:/name/3` → (name, None, "3")."""
    ref = uri.removeprefix("models:/")
    if "@" in ref:
        name, alias = ref.split("@", 1)
        if name and alias:
            return name, alias, None
    elif "/" in ref:
        name, version = ref.split("/", 1)
        if name and version.isdigit():
            return name, None, version
    raise IntentModelUnavailableError(
        f"unsupported model URI {uri!r}: use models:/<name>@<alias> or models:/<name>/<version>"
    )


def resolve_model_uri(uri: str, cache_dir: Path, tracking_uri: str) -> tuple[Path, str]:
    """Local bundle directory and model id for `uri`.

    A plain path is used as is. A `models:/` URI resolves its alias to a version on the
    MLflow server and downloads that version once into `cache_dir/<server>/<name>/v<version>`
    (`<server>` is a hash of the tracking URI, so version numbers of different servers never
    collide); later loads of the same version read the cache. The alias is resolved once per
    process (the model is a singleton), so promoting a new champion takes effect on restart.
    """
    if not uri.startswith("models:/"):
        path = Path(uri).expanduser()
        return path, f"local/{path.name}"

    name, alias, version = parse_registry_uri(uri)
    mlflow = _import_ml("mlflow")
    client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    try:
        version = version or str(client.get_model_version_by_alias(name, alias).version)
        server = hashlib.sha256(tracking_uri.encode()).hexdigest()[:12]
        target = cache_dir / server / name / f"v{version}"
        bundle = target / "artifacts" / BUNDLE_ARTIFACT
        if not (bundle / META_FILE).is_file():
            _download_version(mlflow, client, name, version, target, tracking_uri)
    except IntentModelUnavailableError:
        raise
    except Exception as exc:
        # MLflow errors can be long and quote server responses; the type is enough to act on.
        raise IntentModelUnavailableError(
            f"cannot resolve {uri} from MLflow at {tracking_uri} ({type(exc).__name__})"
        ) from None
    return bundle, f"{name}/v{version}"


def _download_version(
    mlflow: ModuleType, client: Any, name: str, version: str, target: Path, tracking_uri: str
) -> None:
    """Download into a temporary sibling, then rename, so a crash never leaves a half cache."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".download-", dir=target.parent))
    try:
        source = client.get_model_version_download_uri(name, version)
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=source, dst_path=str(staging), tracking_uri=tracking_uri
        )
        if target.exists():
            shutil.rmtree(target)
        Path(downloaded).rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


_lock = threading.Lock()


@cache
def _cached_model(uri: str, cache_dir: Path, device: str | None) -> IntentModel:
    directory, model_id = resolve_model_uri(uri, cache_dir, get_settings().mlflow_tracking_uri)
    return load_bundle(directory, model_id, device)


def get_intent_model(
    uri: str, cache_dir: Path = DEFAULT_CACHE_DIR, device: str | None = None
) -> IntentModel:
    """Shared model for `uri`, loaded on first call; the lock prevents duplicate loads.

    Failures are not cached (functools.cache only stores returns), so a later call retries,
    e.g. once the MLflow server is up.
    """
    with _lock:
        return _cached_model(uri, cache_dir, device)
