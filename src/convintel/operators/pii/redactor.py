"""PII redaction operator: the boundary after which no component sees raw PII."""

import asyncio
import time
from collections.abc import Sequence

from opentelemetry import trace
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult as AnonymizerSpan

from convintel.contracts import Conversation, OperatorResult, PIIEntity, PIIOutput, Turn
from convintel.errors import OperatorError

from .engine import DEFAULT_SPACY_MODEL, SUPPORTED_ENTITIES, get_analyzer

_tracer = trace.get_tracer(__name__)

DEFAULT_SCORE_THRESHOLD = 0.35
"""Keeps Presidio's unvalidated phone matches (0.4) and drops bare account-like numbers (0.05).

Redaction favours recall: a false positive costs a few words of context for downstream
operators, a false negative leaks PII to an LLM provider.
"""


class PIIRedactor:
    """Implements `Operator[PIIOutput]` with Presidio, replacing spans by `<ENTITY_TYPE>`.

    Entity offsets in the output refer to the original turn text, so they can be audited
    against the source; the redacted conversation keeps the turn structure unchanged.
    """

    name = "pii"
    version = "1.0.0"

    def __init__(
        self,
        analyzer: AnalyzerEngine | None = None,
        spacy_model: str = DEFAULT_SPACY_MODEL,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self._analyzer = analyzer
        self._spacy_model = spacy_model
        self._score_threshold = score_threshold
        self._anonymizer = AnonymizerEngine()

    @property
    def model_id(self) -> str:
        return f"presidio/{self._spacy_model}"

    async def run(self, conversation: Conversation) -> OperatorResult[PIIOutput]:
        start = time.perf_counter()
        with _tracer.start_as_current_span(f"operator.{self.name}") as span:
            span.set_attribute("conversation.id", conversation.id)
            span.set_attribute("operator.version", self.version)
            span.set_attribute("model.id", self.model_id)
            try:
                output = await asyncio.to_thread(self._redact, conversation)
            except (OSError, ValueError) as exc:
                raise OperatorError(self.name, f"{type(exc).__name__}: {exc}") from exc
            span.set_attribute("pii.entity_count", len(output.entities))

        return OperatorResult[PIIOutput](
            operator=self.name,
            operator_version=self.version,
            output=output,
            latency_ms=(time.perf_counter() - start) * 1000,
            model_id=self.model_id,
        )

    def _redact(self, conversation: Conversation) -> PIIOutput:
        """Blocking part, run in a worker thread so the event loop keeps serving requests."""
        analyzer = self._analyzer or get_analyzer(self._spacy_model)
        entities: list[PIIEntity] = []
        turns: list[Turn] = []
        for turn in conversation.turns:
            found = analyzer.analyze(
                turn.text,
                language=conversation.language,
                entities=list(SUPPORTED_ENTITIES),
                score_threshold=self._score_threshold,
            )
            spans = merge_overlaps(found)
            entities.extend(
                PIIEntity(
                    entity_type=s.entity_type,
                    turn_index=turn.index,
                    start=s.start,
                    end=s.end,
                    score=s.score,
                )
                for s in spans
            )
            turns.append(turn.model_copy(update={"text": self._anonymize(turn.text, spans)}))

        redacted = conversation.model_copy(update={"turns": tuple(turns), "redacted": True})
        return PIIOutput(entities=entities, redacted_conversation=redacted)

    def _anonymize(self, text: str, spans: Sequence[RecognizerResult]) -> str:
        if not spans:
            return text
        results = [AnonymizerSpan(s.entity_type, s.start, s.end, s.score) for s in spans]
        return self._anonymizer.anonymize(text=text, analyzer_results=results).text


def merge_overlaps(results: Sequence[RecognizerResult]) -> list[RecognizerResult]:
    """Collapse overlapping detections into one span covering their union.

    Presidio's anonymizer drops the weaker of two overlapping spans, which leaks the part of
    the weaker one that sticks out (e.g. an account number regex covering more digits than
    the phone match inside it). The union is always redacted; the label and score come from
    the strongest detection.
    """
    merged: list[RecognizerResult] = []
    for result in sorted(results, key=lambda r: (r.start, -r.end)):
        last = merged[-1] if merged else None
        if last is None or result.start >= last.end:
            span = RecognizerResult(result.entity_type, result.start, result.end, result.score)
            merged.append(span)
            continue
        if result.score > last.score:
            last.entity_type, last.score = result.entity_type, result.score
        last.end = max(last.end, result.end)
    return merged
