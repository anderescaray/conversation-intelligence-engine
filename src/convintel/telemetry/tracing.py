"""OpenTelemetry setup and the `traced` helper every operator uses.

`setup()` exports spans to Phoenix over OTLP/HTTP and turns on OpenInference
auto-instrumentation for LangChain/LangGraph, so `llm.call` spans appear without
extra work in the LLM client. `traced("operator.<name>")` is both a decorator
(sync and async) and a context manager: it wraps the block in a span, records
latency into `convintel_operator_latency_seconds` and failures into
`convintel_operator_errors_total`, then re-raises.

Span content: OpenInference captures LLM inputs/outputs on spans by default,
which is what makes local debugging useful. Everything reaching an LLM call has
already passed PII redaction, so this is safe by design — but if you ever need
to hide payloads (e.g. sharing a Phoenix instance), set OpenInference's own env
vars before startup: OPENINFERENCE_HIDE_INPUTS / OPENINFERENCE_HIDE_OUTPUTS, or
the finer-grained OPENINFERENCE_HIDE_INPUT_MESSAGES,
OPENINFERENCE_HIDE_OUTPUT_MESSAGES, OPENINFERENCE_HIDE_INPUT_TEXT and
OPENINFERENCE_HIDE_OUTPUT_TEXT. See infra/README.md.
"""

import functools
import inspect
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from types import TracebackType
from typing import Any, cast

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span, Status, StatusCode

from convintel.config import Settings, get_settings
from convintel.telemetry.metrics import OPERATOR_ERRORS_TOTAL, OPERATOR_LATENCY_SECONDS

_SERVICE_NAME = "convintel"
_OPERATOR_PREFIX = "operator."

_logger = logging.getLogger(__name__)
_provider: TracerProvider | None = None


def setup(
    *,
    settings: Settings | None = None,
    exporter: SpanExporter | None = None,
) -> TracerProvider:
    """Configure the global tracer provider and OpenInference instrumentation.

    Idempotent: later calls return the provider built by the first. `exporter`
    exists for tests, which inject an in-memory exporter instead of opening a
    connection to Phoenix.
    """
    global _provider
    if _provider is not None:
        return _provider

    # Imported here so `traced` (and tests) don't pay for wrapt/langchain patching.
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.semconv.resource import ResourceAttributes

    settings = settings or get_settings()
    provider = TracerProvider(
        resource=Resource(
            {
                "service.name": _SERVICE_NAME,
                # Phoenix groups spans into projects by this attribute.
                ResourceAttributes.PROJECT_NAME: _SERVICE_NAME,
            }
        )
    )

    if exporter is None:
        endpoint = f"{settings.phoenix_collector_endpoint.rstrip('/')}/v1/traces"
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    else:
        processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    try:
        LangChainInstrumentor().instrument(tracer_provider=provider)
    except Exception:
        # Tracing must never take the app down: spans still flow, just without
        # LangChain auto-instrumentation.
        _logger.warning("LangChain OpenInference instrumentation failed", exc_info=True)
    _provider = provider
    return provider


class Traced(AbstractContextManager[Span]):
    """Result of `traced("operator.<name>")`: decorator (sync/async) and context manager."""

    def __init__(self, name: str, *, version: str | None = None) -> None:
        self.name = name
        self.operator = name.removeprefix(_OPERATOR_PREFIX)
        self._version = version
        self._tracer = trace.get_tracer("convintel.telemetry")
        self._span_cm: AbstractContextManager[Span] | None = None

    def _resolve_version(self, args: tuple[object, ...]) -> str:
        # Decorating a method defaults `version` to the instance's own `version`
        # attribute, which every Operator implementation exposes per the protocol.
        if self._version is not None:
            return self._version
        if args and isinstance(v := getattr(args[0], "version", None), str):
            return v
        return "unknown"

    def _observe(self, version: str, exc: BaseException | None, elapsed: float) -> None:
        OPERATOR_LATENCY_SECONDS.labels(operator=self.operator, version=version).observe(elapsed)
        if exc is not None:
            OPERATOR_ERRORS_TOTAL.labels(
                operator=self.operator, error_type=type(exc).__name__
            ).inc()

    @contextmanager
    def _span_and_metrics(self, version: str) -> Iterator[Span]:
        # record_exception=False and set_status_on_exception=False are deliberate:
        # the defaults attach str(exc) to the span, and exception messages can
        # carry PII. We only ever write type(exc).__name__ (error.type, and the
        # error_type metric label), never the message.
        start = time.perf_counter()
        with self._tracer.start_as_current_span(
            self.name, record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("operator.version", version)
            try:
                yield span
            except BaseException as exc:
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR))
                self._observe(version, exc, time.perf_counter() - start)
                raise
        self._observe(version, None, time.perf_counter() - start)

    def __call__[F: Callable[..., Any]](self, fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self._span_and_metrics(self._resolve_version(args)):
                    return await fn(*args, **kwargs)

            return cast(F, async_wrapper)

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with self._span_and_metrics(self._resolve_version(args)):
                return fn(*args, **kwargs)

        return cast(F, sync_wrapper)

    def __enter__(self) -> Span:
        self._span_cm = self._span_and_metrics(self._resolve_version(()))
        return self._span_cm.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._span_cm is not None:
            self._span_cm.__exit__(exc_type, exc_value, traceback)
            self._span_cm = None


def traced(name: str, *, version: str | None = None) -> Traced:
    """Wrap a callable or block in an OTel span and record operator metrics.

    Usage: `@traced("operator.intent")` on a function (sync or async), or
    `with traced("operator.intent"):`. The metric `operator` label is `name`
    minus the `operator.` prefix.
    """
    return Traced(name, version=version)
