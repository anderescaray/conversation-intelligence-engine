"""Request accounting and the last-resort 500, as one pure ASGI middleware.

It sits outside FastAPI's exception handlers, so it sees the final status of every request,
and inside Starlette's `ServerErrorMiddleware`, whose default would log the traceback (and the
exception message, which may quote conversation text) before answering.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from convintel.api.errors import INTERNAL_ERROR
from convintel.telemetry import REQUESTS_TOTAL

UNMATCHED_ROUTE = "unmatched"

_logger = logging.getLogger(__name__)


@contextmanager
def api_span(name: str) -> Iterator[Span]:
    """Span that records only `error.type` on failure, never the exception message."""
    with trace.get_tracer("convintel.api").start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as span:
        try:
            yield span
        except BaseException as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise


def _route(scope: Scope) -> str:
    # The template ("/v1/conversations"), never the raw path, keeps the label bounded.
    return getattr(scope.get("route"), "path", None) or UNMATCHED_ROUTE


class RequestMetricsMiddleware:
    def __init__(self, app: ASGIApp, *, excluded_routes: frozenset[str] = frozenset()) -> None:
        self.app = app
        self.excluded_routes = excluded_routes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        status = 500
        started = False

        async def send_and_record(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                status, started = message["status"], True
            await send(message)

        try:
            await self.app(scope, receive, send_and_record)
        except Exception as exc:
            _logger.error("Unhandled %s on %s", type(exc).__name__, _route(scope))
            status = 500
            if started:
                # Headers are on the wire, so no 500 body can follow and the server must abort
                # the response; it still counts as a 500 whatever status was already sent.
                raise
            await JSONResponse({"detail": INTERNAL_ERROR}, status_code=500)(scope, receive, send)
        finally:
            route = _route(scope)
            if route not in self.excluded_routes:
                REQUESTS_TOTAL.labels(route=route, status=str(status)).inc()
