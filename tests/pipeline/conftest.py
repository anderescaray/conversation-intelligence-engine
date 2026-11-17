from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from convintel.pipeline import pipeline as pipeline_module


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """Capture the spans of the pipeline root, traced() and the operators in memory.

    The process-wide tracer provider can be set only once, so the tracers are patched instead:
    ``trace.get_tracer`` for the ones created per call, and the pipeline's module-level one.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    monkeypatch.setattr(pipeline_module, "_tracer", provider.get_tracer(pipeline_module.__name__))
    yield exporter
    provider.shutdown()
