"""Map failures to responses whose bodies are fixed strings.

Exception messages may quote conversation text (pydantic echoes input, operators may include
snippets), so no handler ever returns `str(exc)` or the offending input.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from convintel.api.audio import AudioRejectedError
from convintel.errors import ConvIntelError, LLMError, OperatorError, TranscriptionError

INTERNAL_ERROR = "Internal server error."

_logger = logging.getLogger(__name__)


class ServiceNotConfiguredError(ConvIntelError):
    """A dependency the route needs (pipeline, Twilio credentials, fetcher) is not configured."""


# Most specific first: the first isinstance match wins.
_FAILURES: tuple[tuple[type[ConvIntelError], int, str], ...] = (
    (ServiceNotConfiguredError, 503, "Service not configured."),
    (TranscriptionError, 500, "Transcription failed."),
    (LLMError, 502, "An upstream model call failed."),
    (OperatorError, 500, "Conversation analysis failed."),
    (ConvIntelError, 500, "The request could not be processed."),
)


async def _domain_error(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, AudioRejectedError):
        status = 413 if exc.reason in {"too_large", "too_long"} else 422
        return JSONResponse({"detail": "Audio rejected.", "reason": exc.reason}, status_code=status)
    status, message = next((s, m) for t, s, m in _FAILURES if isinstance(exc, t))
    _logger.warning("Request failed with %s (%d)", type(exc).__name__, status)
    return JSONResponse({"detail": message}, status_code=status)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    return JSONResponse(
        {
            "detail": "Invalid request body.",
            "errors": [{"loc": list(e.get("loc", ())), "type": e.get("type")} for e in errors],
        },
        status_code=422,
    )


def install(app: FastAPI) -> None:
    app.add_exception_handler(ConvIntelError, _domain_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
