"""Intent operator: calibrated Banking77 intent of the customer's most confident turn."""

import asyncio
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from convintel.contracts import Conversation, IntentOutput, OperatorResult, Speaker
from convintel.errors import OperatorError
from convintel.telemetry import traced

from .bundle import IntentModelUnavailableError
from .calibration import softmax
from .model import DEFAULT_CACHE_DIR, DEFAULT_MODEL_URI, IntentModel, get_intent_model

UNKNOWN_LABEL = "unknown"
CLASSIFIER_SOURCE = "classifier"
DEFAULT_TOP_K = 3
_WARMUP_TEXT = "I would like to check my card balance"

_inference_lock = threading.Lock()
"""Serializes inference across worker threads.

Every operator shares the process-wide model, and a torch module plus its tokenizer are not
documented as safe to call concurrently. One batch at a time per process; scale with worker
processes (same pattern as the sentiment operator).
"""


def pick_intent(
    probs: NDArray[np.floating], labels: Sequence[str], top_k: int = DEFAULT_TOP_K
) -> IntentOutput:
    """Intent of the turn whose calibrated top-1 probability is highest.

    Banking77 is single-utterance, so each customer turn is scored on its own and the clearest
    statement of the request wins; greetings and "yes, that's right" score low and lose.

    Selection bias: the max over several calibrated confidences is optimistically biased
    (each turn is calibrated, their maximum is not), more so the more customer turns there are.
    τ was chosen per utterance on Banking77; it must be re-tuned at conversation level on the
    T06 golden set before the pipeline routes on it.
    """
    turn = int(probs.max(axis=1).argmax())
    row = probs[turn]
    order = np.argsort(row, kind="stable")[::-1][:top_k]
    ranked = [(labels[int(i)], min(1.0, float(row[i]))) for i in order]
    return IntentOutput(
        label=ranked[0][0], confidence=ranked[0][1], source=CLASSIFIER_SOURCE, top_k=ranked
    )


@contextmanager
def _as_operator_error(operator: str) -> Iterator[None]:
    try:
        yield
    except OperatorError:
        raise
    except IntentModelUnavailableError as exc:
        # Built by us from paths and URIs, never from conversation text: safe to surface.
        raise OperatorError(operator, str(exc)) from None
    except Exception as exc:
        # Third-party messages (tokenizer, torch, MLflow) can quote input text. Keep only the
        # type and cut the chain: span exception events format the whole traceback, causes
        # included. Same policy as the PII and sentiment operators.
        raise OperatorError(operator, f"intent inference failed ({type(exc).__name__})") from None


class IntentOperator:
    """Implements `Operator[IntentOutput]` over the customer's turns only.

    Each customer turn is classified in one batch; `pick_intent` keeps the most confident one
    (see its docstring for the selection bias this introduces). A conversation without customer
    turns returns `label="unknown"`, confidence 0 and does not load the model; the pipeline
    must not send it to the LLM verifier, as there is nothing to verify.

    The model is resolved from `model_uri` (`models:/name@alias`, `models:/name/<version>` or a
    local bundle directory). Call `warmup()` at startup so the first conversation does not pay
    for resolution and loading; `run()` loads lazily if it was not called.
    """

    name = "intent"
    version = "1.0.0"

    def __init__(
        self,
        model: IntentModel | None = None,
        model_uri: str = DEFAULT_MODEL_URI,
        top_k: int = DEFAULT_TOP_K,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        device: str | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._model = model
        self._model_uri = model_uri
        self._top_k = top_k
        self._cache_dir = cache_dir
        self._device = device

    @property
    def threshold(self) -> float | None:
        """τ shipped with the loaded model (None until loaded). The graph routes on it."""
        return self._model.threshold if self._model is not None else None

    async def warmup(self) -> None:
        """Resolve, load and run one inference now, so startup fails fast if the model is
        missing and the first request does not pay for loading or CUDA initialization."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            f"operator.{self.name}.warmup", record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("operator.version", self.version)
            try:
                with _as_operator_error(self.name):
                    model, _ = await asyncio.to_thread(self._predict, [_WARMUP_TEXT])
            except OperatorError as exc:
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise
            span.set_attribute("model.id", model.model_id)

    async def run(self, conversation: Conversation) -> OperatorResult[IntentOutput]:
        start = time.perf_counter()
        texts = [t.text for t in conversation.turns if t.speaker is Speaker.CUSTOMER]
        model_id: str | None = None
        with traced(f"operator.{self.name}", version=self.version) as span:
            span.set_attribute("conversation.id", conversation.id)
            span.set_attribute("intent.customer_turns", len(texts))
            if texts:
                with _as_operator_error(self.name):
                    model, probs = await asyncio.to_thread(self._predict, texts)
                output = pick_intent(probs, model.labels, self._top_k)
                model_id = model.model_id
                span.set_attribute("model.id", model_id)
            else:
                output = IntentOutput(
                    label=UNKNOWN_LABEL, confidence=0.0, source=CLASSIFIER_SOURCE, top_k=[]
                )
            span.set_attribute("intent.label", output.label)
            span.set_attribute("intent.confidence", output.confidence)

        return OperatorResult[IntentOutput](
            operator=self.name,
            operator_version=self.version,
            output=output,
            latency_ms=(time.perf_counter() - start) * 1000,
            model_id=model_id,
        )

    def _predict(self, texts: list[str]) -> tuple[IntentModel, NDArray[np.float64]]:
        if self._model is None:
            self._model = get_intent_model(self._model_uri, self._cache_dir, self._device)
        model = self._model
        with _inference_lock:
            logits = model.predict_logits(texts)
        return model, softmax(logits, model.temperature)
