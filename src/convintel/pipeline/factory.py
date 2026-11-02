"""Builds the production pipeline from settings.

Building is cheap and does no I/O: every model is loaded, and the LLM config validated, in
``Pipeline.warmup()``, which the API awaits at startup. Settings are passed in explicitly, so
the pipeline never falls back to the process-wide ``get_settings()``.
"""

from convintel.config import Settings
from convintel.llm import LLMClient
from convintel.operators.intent import IntentOperator
from convintel.operators.llm import SummaryOperator
from convintel.operators.pii import PIIRedactor
from convintel.operators.sentiment import SentimentOperator

from .pipeline import Pipeline


def build_pipeline(settings: Settings) -> Pipeline:
    """The pipeline with the real operators. Call ``warmup()`` before ``analyze()``."""
    return Pipeline(
        redactor=PIIRedactor(),
        sentiment=SentimentOperator(),
        summary=SummaryOperator(client=LLMClient(settings)),
        intent=IntentOperator(),
        warmup_timeout_s=settings.pipeline_warmup_timeout_s,
        threshold_override=settings.intent_verifier_threshold,
    )
