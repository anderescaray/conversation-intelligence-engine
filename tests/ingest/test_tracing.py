import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from convintel.errors import ConvIntelError
from convintel.ingest.normalizers import normalize_chat


def test_trace_does_not_record_raw_validation_input(spans: InMemorySpanExporter) -> None:
    with pytest.raises(ConvIntelError):
        normalize_chat("private-synthetic-input")
    (span,) = spans.get_finished_spans()
    assert span.name == "ingest.chat.normalize"
    assert span.status.status_code is StatusCode.ERROR
    assert span.events == ()
    assert "private-synthetic-input" not in str(span.attributes)
    assert not span.status.description
