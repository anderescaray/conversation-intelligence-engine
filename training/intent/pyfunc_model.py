"""MLflow pyfunc wrapper around the intent bundle (logged as "models from code").

The registry needs a logged model to version. The operator never goes through pyfunc: it
downloads the `bundle` artifact and loads it with `convintel.operators.intent.load_bundle`.
The wrapper exists so the registered model is also directly usable with `mlflow.pyfunc`
(e.g. for ad-hoc scoring); it needs the convintel package in that environment.
"""

import importlib
from pathlib import Path
from typing import Any

from convintel.operators.intent import load_bundle, softmax

_mlflow = importlib.import_module("mlflow")


class IntentPyfunc(_mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        self._model = load_bundle(Path(context.artifacts["bundle"]), model_id="pyfunc")

    def predict(self, context: Any, model_input: list[str], params: Any = None) -> list[str]:
        """Calibrated top-1 label per text."""
        probs = softmax(self._model.predict_logits(model_input), self._model.temperature)
        return [self._model.labels[i] for i in probs.argmax(axis=1)]


_mlflow.models.set_model(IntentPyfunc())
