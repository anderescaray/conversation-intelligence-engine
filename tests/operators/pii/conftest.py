from collections.abc import Iterator
from pathlib import Path

import pytest
import spacy
import tldextract
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pii_synthetic_cases import SYNTHETIC_NAMES
from presidio_analyzer import AnalyzerEngine, PatternRecognizer

from convintel.operators.pii import PIIRedactor, build_analyzer


@pytest.fixture(scope="session", autouse=True)
def _offline_tldextract() -> Iterator[None]:
    """Presidio's email validator fetches the public suffix list on first use; use the snapshot.

    Session-scoped, so it patches by hand instead of through the function-scoped monkeypatch.
    """
    original = tldextract.extract
    tldextract.extract = tldextract.TLDExtract(suffix_list_urls=())
    yield
    tldextract.extract = original


@pytest.fixture(scope="session")
def blank_spacy_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tokenizer-only English pipeline saved to disk: no download, no NER."""
    path = tmp_path_factory.mktemp("spacy") / "blank_en"
    spacy.blank("en").to_disk(path)
    return path


@pytest.fixture(scope="session")
def analyzer(blank_spacy_model: Path) -> AnalyzerEngine:
    """Real Presidio analyzer whose PERSON detection is a deny-list instead of spaCy NER."""
    fake_ner = PatternRecognizer(
        supported_entity="PERSON",
        deny_list=SYNTHETIC_NAMES,
        deny_list_score=0.85,
        name="FakePersonNER",
    )
    return build_analyzer(str(blank_spacy_model), extra_recognizers=[fake_ner])


@pytest.fixture
def redactor(analyzer: AnalyzerEngine) -> PIIRedactor:
    return PIIRedactor(analyzer=analyzer, spacy_model="blank_en")


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    yield exporter
    provider.shutdown()
