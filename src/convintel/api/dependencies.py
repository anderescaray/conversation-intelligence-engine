"""Per-app services, reached by routes only through FastAPI dependencies.

`create_app` stores one `ApiState` on `app.state`; tests inject fakes there or through
`app.dependency_overrides`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Request

from convintel.api.audio import AUDIO_SPOOL_TTL_S, AudioSpool
from convintel.api.errors import ServiceNotConfiguredError
from convintel.api.protocols import Pipeline, RecordingFetcher
from convintel.config import Settings, get_settings
from convintel.contracts import Transcriber


@dataclass
class ApiState:
    pipeline: Pipeline | None
    transcriber: Transcriber
    spool: AudioSpool
    telemetry_setup: Callable[[], object]
    fetcher: RecordingFetcher | None = None
    settings: Settings | None = None
    public_base_url: str | None = None
    spool_ttl_s: float = AUDIO_SPOOL_TTL_S
    ready: bool = field(default=False, init=False)


def get_state(request: Request) -> ApiState:
    return request.app.state.convintel


State = Annotated[ApiState, Depends(get_state)]


def get_pipeline(state: State) -> Pipeline:
    if state.pipeline is None:
        raise ServiceNotConfiguredError("No pipeline configured.")
    return state.pipeline


def get_transcriber(state: State) -> Transcriber:
    return state.transcriber


def get_spool(state: State) -> AudioSpool:
    return state.spool


def get_fetcher(state: State) -> RecordingFetcher:
    if state.fetcher is None:
        raise ServiceNotConfiguredError("No recording fetcher configured.")
    return state.fetcher


def get_app_settings(state: State) -> Settings:
    # Read at request time, never at import time: importing the API parses no env. In
    # production `create_production_app()` (served with `--factory`) passes its Settings in.
    return state.settings or get_settings()
