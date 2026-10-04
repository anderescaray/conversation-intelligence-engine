"""Tests for convintel.telemetry.setup with an in-memory exporter (no network).

The global tracer provider can only be set once per process, so a single
module-scoped setup is shared and the exporter is cleared between tests.
"""

import time
from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from convintel.config import Settings
from convintel.telemetry import traced, tracing
from convintel.telemetry.tracing import setup


@pytest.fixture(scope="module")
def exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = setup(settings=Settings(_env_file=None), exporter=exporter)  # pyright: ignore[reportCallIssue]
    yield exporter
    provider.shutdown()


@pytest.fixture(autouse=True)
def _clear_spans(exporter: InMemorySpanExporter) -> None:
    exporter.clear()


def test_setup_is_idempotent(exporter: InMemorySpanExporter) -> None:
    assert setup(exporter=exporter) is setup()


def test_setup_sets_global_provider() -> None:
    span = trace.get_tracer("t").start_span("probe")
    assert span.get_span_context().trace_id != 0


def test_traced_spans_are_exported(exporter: InMemorySpanExporter) -> None:
    with traced("operator.fake", version="9.9.9"):
        pass
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["operator.fake"]
    assert spans[0].attributes is not None
    assert spans[0].attributes["operator.version"] == "9.9.9"


def test_spans_nest_as_children(exporter: InMemorySpanExporter) -> None:
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("conversation.process"), traced("operator.fake"):
        pass
    root, child = exporter.get_finished_spans()[1], exporter.get_finished_spans()[0]
    assert root.name == "conversation.process"
    assert child.name == "operator.fake"
    assert child.parent is not None
    root_ctx = root.get_span_context()
    assert root_ctx is not None
    assert child.parent.span_id == root_ctx.span_id


def test_exception_message_never_reaches_the_span(exporter: InMemorySpanExporter) -> None:
    """Exception messages can carry PII; only type(exc).__name__ may be recorded."""
    pii = "customer John Smith, card 4111-1111-1111-1111"

    class PiiBearingError(Exception):
        pass

    with pytest.raises(PiiBearingError), traced("operator.probe"):
        raise PiiBearingError(pii)

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert not span.status.description or pii not in span.status.description
    assert span.attributes is not None
    assert span.attributes["error.type"] == "PiiBearingError"
    assert all(pii not in str(value) for value in span.attributes.values())
    for event in span.events:
        assert pii not in event.name
        assert all(pii not in str(v) for v in (event.attributes or {}).values())


def test_setup_survives_instrumentation_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing LangChainInstrumentor must warn, not crash the app."""
    import openinference.instrumentation.langchain

    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(
        openinference.instrumentation.langchain.LangChainInstrumentor,
        "instrument",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("patch failed")),
    )

    provider = setup(exporter=InMemorySpanExporter())

    assert isinstance(provider, TracerProvider)
    assert "instrumentation failed" in caplog.text.lower()


def test_traced_does_not_block_on_unreachable_collector() -> None:
    """With the OTLP endpoint down, spans are queued/dropped, never blocking."""
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint="http://127.0.0.1:1/v1/traces"))
    )
    span_source = traced("operator.probe")
    span_source._tracer = provider.get_tracer("test")  # bypass the global provider

    start = time.monotonic()
    with span_source:
        pass
    elapsed = time.monotonic() - start

    provider.shutdown()
    assert elapsed < 2.0
