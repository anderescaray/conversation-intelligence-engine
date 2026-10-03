"""Presidio analyzer construction and the process-wide lazy singleton.

Loading the spaCy pipeline takes seconds and hundreds of MB, so it happens once per process
and per model, on first use rather than at import (importing the package must stay cheap for
the API, CLI and tests). Presidio downloads the spaCy model itself if it is not installed.
"""

import threading
from collections.abc import Sequence
from functools import cache

from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import SpacyNlpEngine

from .recognizers import ACCOUNT_NUMBER, AccountNumberRecognizer

DEFAULT_SPACY_MODEL = "en_core_web_lg"
"""Presidio's recommended English model; `sm` misses noticeably more PERSON entities."""

SUPPORTED_ENTITIES: tuple[str, ...] = (
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "IBAN_CODE",
    ACCOUNT_NUMBER,
)

_lock = threading.Lock()


def build_analyzer(
    spacy_model: str, extra_recognizers: Sequence[EntityRecognizer] = ()
) -> AnalyzerEngine:
    """Analyzer with Presidio's predefined recognizers plus ours, English only.

    `spacy_model` is a package name or a path to a saved pipeline. `extra_recognizers` lets
    tests stand in for the NER model without downloading it.
    """
    nlp_engine = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": spacy_model}])
    nlp_engine.load()
    registry = RecognizerRegistry(supported_languages=["en"])
    registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en"])
    registry.add_recognizer(AccountNumberRecognizer())
    for recognizer in extra_recognizers:
        registry.add_recognizer(recognizer)
    return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["en"])


@cache
def _cached_analyzer(spacy_model: str) -> AnalyzerEngine:
    return build_analyzer(spacy_model)


def get_analyzer(spacy_model: str = DEFAULT_SPACY_MODEL) -> AnalyzerEngine:
    """Shared analyzer for `spacy_model`, built on first call.

    The lock makes concurrent first calls (several conversations arriving at startup) build
    the model once instead of racing; `functools.cache` alone does not guarantee that.
    """
    with _lock:
        return _cached_analyzer(spacy_model)
