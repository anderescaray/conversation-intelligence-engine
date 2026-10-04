"""Typed training config, read from `config.yaml` so every run is reproducible from one file."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_model: str
    max_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    eval_batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    warmup_ratio: float = Field(ge=0, lt=1)
    epochs: int = Field(gt=0)
    grad_clip: float = Field(gt=0)
    bf16: bool
    seed: int
    val_fraction: float = Field(gt=0, lt=1)
    ece_bins: int = Field(gt=0)
    target_covered_accuracy: float = Field(gt=0, le=1)
    min_covered_cases: int = Field(gt=0)
    promote_min_macro_f1: float = Field(ge=0, le=1)
    experiment: str
    registered_model: str


def load_config(path: Path = DEFAULT_CONFIG) -> TrainConfig:
    return TrainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
