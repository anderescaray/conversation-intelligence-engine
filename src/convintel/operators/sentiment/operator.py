"""Sentiment operator: per-turn customer sentiment plus a recency-weighted overall label."""

import asyncio
import time

from opentelemetry import trace

from convintel.contracts import Conversation, OperatorResult, SentimentOutput, Speaker
from convintel.errors import OperatorError

from .model import (
    DEFAULT_MODEL_ID,
    LabelMappingError,
    SentimentModel,
    SentimentScores,
    get_sentiment_model,
)
from .trajectory import overall_sentiment

_tracer = trace.get_tracer(__name__)


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
        with _tracer.start_as_current_span(f"operator.{self.name}") as span:
            span.set_attribute("conversation.id", conversation.id)
            span.set_attribute("operator.version", self.version)
            span.set_attribute("model.id", self._model_id)
            span.set_attribute("sentiment.customer_turns", len(texts))
            try:
                scores = await asyncio.to_thread(self._predict, texts)
            except (OSError, LabelMappingError) as exc:
                raise OperatorError(self.name, f"{type(exc).__name__}: {exc}") from exc
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
        return model.predict(texts)
