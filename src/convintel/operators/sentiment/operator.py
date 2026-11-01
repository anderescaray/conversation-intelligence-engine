"""Sentiment operator: per-turn customer sentiment plus a recency-weighted overall label."""

import asyncio
import threading
import time

from opentelemetry import trace

from convintel.contracts import Conversation, OperatorResult, SentimentOutput, Speaker
from convintel.errors import OperatorError

from .model import (
    DEFAULT_MODEL_ID,
    SentimentModel,
    SentimentScores,
    get_sentiment_model,
)
from .trajectory import overall_sentiment

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
    """

    name = "sentiment"
    version = "1.0.0"

    def __init__(self, model: SentimentModel | None = None, model_id: str = DEFAULT_MODEL_ID):
        self._model = model
        self._model_id = model.model_id if model is not None else model_id

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

    def _predict(self, texts: list[str]) -> list[SentimentScores]:
        if not texts:
            return []
        model = self._model or get_sentiment_model(self._model_id)
        with _inference_lock:
            return model.predict(texts)
