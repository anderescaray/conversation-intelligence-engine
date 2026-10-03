"""Tests for convintel.telemetry.setup with an in-memory exporter (no network).

The global tracer provider can only be set once per process, so a single
module-scoped setup is shared and the exporter is cleared between tests.
"""

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from convintel.config import Settings
from convintel.telemetry import traced
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
