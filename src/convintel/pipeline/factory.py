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

from .graph import IntentVerifier
from .pipeline import KeepClassifierLabel, Pipeline
from .verifier import LLMIntentVerifier

VERIFIER_TOP_K = 5
"""Candidates the classifier hands to the verifier when it is enabled; see
training/intent/reports/verifier-candidates.md."""


def build_pipeline(settings: Settings) -> Pipeline:
    """The pipeline with the real operators. Call ``warmup()`` before ``analyze()``.

    The LLM verifier, and the top-5 candidates it needs, are plugged in only with
    ``INTENT_VERIFIER_ENABLED``. Until the verifier passes its evaluation gate the flag is off
    and the intent output is exactly the one before the verifier existed: the classifier's
    label and its default ``top_k``.
    """
    client = LLMClient(settings)
    if settings.intent_verifier_enabled:
        intent = IntentOperator(top_k=VERIFIER_TOP_K)
        verify: IntentVerifier = LLMIntentVerifier(client)
    else:
        intent = IntentOperator()
        verify = KeepClassifierLabel()
    return Pipeline(
        redactor=PIIRedactor(),
        sentiment=SentimentOperator(),
        summary=SummaryOperator(client=client),
        intent=intent,
        warmup_timeout_s=settings.pipeline_warmup_timeout_s,
        verify=verify,
        threshold_override=settings.intent_verifier_threshold,
    )
