"""FastAPI application factory and the module-level `app` served by uvicorn.

Startup fails loudly instead of serving a half-ready process: telemetry is configured first so
warmup is traced, then the pipeline warms up, and only then does `/readyz` report ready.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from convintel.api import errors
from convintel.api.audio import SPOOL_DIR, AudioSpool
from convintel.api.dependencies import ApiState
from convintel.api.errors import ServiceNotConfiguredError
from convintel.api.middleware import RequestMetricsMiddleware, api_span
from convintel.api.protocols import Pipeline, RecordingFetcher
from convintel.api.routes import router
from convintel.api.twilio import router as twilio_router
from convintel.config import Settings
from convintel.contracts import Transcriber
from convintel.telemetry import setup
from convintel.transcription.whisper import WhisperTranscriber

# Probes and scrapes would dilute the error-rate SLO computed from convintel_requests_total.
PROBE_ROUTES = frozenset({"/healthz", "/readyz", "/metrics"})

_logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    state: ApiState = app.state.convintel
    state.telemetry_setup()
    if state.pipeline is None:
        raise ServiceNotConfiguredError("No pipeline configured; pass one to create_app().")
    with api_span("api.warmup"):
        try:
            await state.pipeline.warmup()
        except Exception as exc:
            _logger.error("Pipeline warmup failed with %s; aborting startup", type(exc).__name__)
            raise
    state.ready = True
    _logger.info("Pipeline warm; ready to serve")
    try:
        yield
    finally:
        state.ready = False


def create_app(
    *,
    pipeline: Pipeline | None = None,
    transcriber: Transcriber | None = None,
    fetcher: RecordingFetcher | None = None,
    spool_dir: Path = SPOOL_DIR,
    settings: Settings | None = None,
    public_base_url: str | None = None,
    telemetry_setup: Callable[[], object] = setup,
) -> FastAPI:
    """Build the app. Every argument is a seam until `Settings` grows the API fields.

    `public_base_url` is the external origin Twilio calls (e.g. `https://api.example.com`);
    without it the request URL as seen by uvicorn is what gets signature-checked.
    """
    app = FastAPI(title="Conversation Intelligence Engine", lifespan=_lifespan)
    app.state.convintel = ApiState(
        pipeline=pipeline,
        transcriber=transcriber or WhisperTranscriber(),
        spool=AudioSpool(spool_dir),
        telemetry_setup=telemetry_setup,
        fetcher=fetcher,
        settings=settings,
        public_base_url=public_base_url,
    )
    errors.install(app)
    app.add_middleware(RequestMetricsMiddleware, excluded_routes=PROBE_ROUTES)
    app.include_router(router)
    app.include_router(twilio_router)
    return app


# No pipeline on main yet (T10): serving this instance fails at startup until it is wired here.
app = create_app()
