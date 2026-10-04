from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace


@contextmanager
def private_span(name: str) -> Iterator[trace.Span]:
    with trace.get_tracer("convintel.ingest_asr").start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as span:
        try:
            yield span
        except Exception:
            span.set_status(trace.StatusCode.ERROR)
            raise
