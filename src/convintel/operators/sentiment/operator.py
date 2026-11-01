"""Sentiment operator: per-turn customer sentiment plus a recency-weighted overall label."""

import asyncio
import threading
import time

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from convintel.contracts import Conversation, OperatorResult, SentimentOutput, Speaker
from convintel.errors import OperatorError

from .model import (
    DEFAULT_MODEL_ID,
    SentimentModel,
    SentimentScores,
    get_sentiment_model,
)
from .trajectory import overall_sentiment

_WARMUP_TEXT = "I would like to check my card balance"
"""Fixed and free of PII: warmup spans and errors can never carry conversation text."""

_inference_lock = threading.Lock()
"""Serializes inference across worker threads.

transformers pipelines are not thread-safe and every operator shares the process-wide model.
One batch at a time per process; scale with worker processes.
"""


class SentimentOperator:
    """Implements `Operator[SentimentOutput]` over the customer's turns only.

    Agent turns are scripted politeness and would pull every conversation towards positive.
    A conversation without customer turns is neutral with an empty trajectory, and does not
    load the model.

    Call `warmup()` at startup so a missing model fails startup instead of the first request;
    `run()` still loads lazily if it was not called.
    """

    name = "sentiment"
    version = "1.0.0"

    def __init__(self, model: SentimentModel | None = None, model_id: str = DEFAULT_MODEL_ID):
        self._model = model
        self._model_id = model.model_id if model is not None else model_id

    async def warmup(self) -> None:
        """Load the shared model and run one inference now, so startup fails fast if the model
        is missing and the first request does not pay for loading or CUDA initialization."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            f"operator.{self.name}.warmup", record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("operator.version", self.version)
            span.set_attribute("model.id", self._model_id)
            try:
                try:
                    model = await asyncio.to_thread(self._warm)
                except OperatorError:
                    raise
                except Exception as exc:
                    # Same policy as run(): third-party messages keep only their type.
                    raise OperatorError(
                        self.name, f"warmup failed ({type(exc).__name__})"
                    ) from None
            except OperatorError as exc:
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise
            # Keep the warmed model: run() must use the model that was checked at startup.
            self._model = model

    async def run(self, conversation: Conversation) -> OperatorResult[SentimentOutput]:
        start = time.perf_counter()
        texts = [t.text for t in conversation.turns if t.speaker is Speaker.CUSTOMER]
        # The caller opens the span with traced() and records latency and errors once; the
        # operator only annotates the current span (a no-op when there is none).
        span = trace.get_current_span()
        span.set_attribute("conversation.id", conversation.id)
        span.set_attribute("model.id", self._model_id)
        span.set_attribute("sentiment.customer_turns", len(texts))
        try:
            scores = await asyncio.to_thread(self._predict, texts)
        except OperatorError:
            raise
        except Exception as exc:
            # Any third-party message (tokenizer, torch, label mapping) can quote turn text.
            # Keep only the type, and cut the chain: the caller's span may format the whole
            # traceback, causes included. Same policy as the PII redactor.
            raise OperatorError(
                self.name, f"sentiment inference failed ({type(exc).__name__})"
            ) from None
        output = SentimentOutput(
            overall=overall_sentiment(scores),
            customer_trajectory=[s.label for s in scores],
        )
        span.set_attribute("sentiment.overall", output.overall.value)

        return OperatorResult[SentimentOutput](
            operator=self.name,
            operator_version=self.version,
            output=output,
            latency_ms=(time.perf_counter() - start) * 1000,
            model_id=self._model_id,
        )

    def _warm(self) -> SentimentModel:
        model = self._model or get_sentiment_model(self._model_id)
        with _inference_lock:
            model.predict([_WARMUP_TEXT])
        return model

    def _predict(self, texts: list[str]) -> list[SentimentScores]:
        if not texts:
            return []
        model = self._model or get_sentiment_model(self._model_id)
        with _inference_lock:
            return model.predict(texts)
