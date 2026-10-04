"""The on-disk intent model bundle: a Hugging Face model directory plus `intent_meta.json`.

Training writes it, MLflow stores it as the `bundle` artifact of the registered model, and the
operator loads it. Calibration (T, τ) travels inside the bundle so a model version can never
be served with another version's temperature.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from convintel.errors import ConvIntelError

META_FILE = "intent_meta.json"


class IntentModelUnavailableError(ConvIntelError):
    """The intent model cannot be loaded: missing bundle, unreachable registry, no `ml` group.

    Messages carry paths and URIs only, never conversation text, so the operator can surface
    them verbatim.
    """


class IntentBundleMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    labels: tuple[str, ...] = Field(min_length=2, description="Class names in logit order")
    temperature: float = Field(gt=0, description="Fitted on the validation split")
    threshold: float | None = Field(
        default=None, ge=0, le=1, description="τ from the validation coverage study"
    )
    max_length: int = Field(gt=0, description="Tokenizer truncation used in training")
    base_model: str
    dataset_version: str


def write_bundle_meta(directory: Path, meta: IntentBundleMeta) -> None:
    (directory / META_FILE).write_text(meta.model_dump_json(indent=2), encoding="utf-8")


def read_bundle_meta(directory: Path) -> IntentBundleMeta:
    path = directory / META_FILE
    if not path.is_file():
        raise IntentModelUnavailableError(
            f"intent model bundle not found at {directory} (no {META_FILE}); "
            "train one with training/intent/train.py or point model_uri at a registered model"
        )
    try:
        return IntentBundleMeta.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        raise IntentModelUnavailableError(f"invalid {META_FILE} in {directory}") from None
