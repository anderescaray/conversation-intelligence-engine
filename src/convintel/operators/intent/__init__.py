"""Customer intent (Banking77 labels) from a fine-tuned, temperature-calibrated DistilBERT."""

from .bundle import META_FILE, IntentBundleMeta, IntentModelUnavailableError, read_bundle_meta
from .calibration import expected_calibration_error, fit_temperature, softmax
from .evaluation import choose_threshold, coverage_accuracy_curve, macro_f1
from .model import (
    DEFAULT_MODEL_URI,
    REGISTERED_MODEL_NAME,
    HFIntentModel,
    IntentModel,
    get_intent_model,
    load_bundle,
)
from .operator import UNKNOWN_LABEL, IntentOperator, pick_intent

__all__ = [
    "DEFAULT_MODEL_URI",
    "META_FILE",
    "REGISTERED_MODEL_NAME",
    "UNKNOWN_LABEL",
    "HFIntentModel",
    "IntentBundleMeta",
    "IntentModel",
    "IntentModelUnavailableError",
    "IntentOperator",
    "choose_threshold",
    "coverage_accuracy_curve",
    "expected_calibration_error",
    "fit_temperature",
    "get_intent_model",
    "load_bundle",
    "macro_f1",
    "pick_intent",
    "read_bundle_meta",
    "softmax",
]
