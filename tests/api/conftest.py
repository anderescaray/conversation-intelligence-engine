import socket
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from api_support import ACCOUNT, TOKEN, FakePipeline, FakeTranscriber
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from convintel.api import create_app
from convintel.config import Settings


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unit tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    yield exporter
    provider.shutdown()


@pytest.fixture
def settings() -> Settings:
    return Settings.model_validate(
        {"_env_file": None, "twilio_auth_token": SecretStr(TOKEN), "twilio_account_sid": ACCOUNT}
    )


@pytest.fixture
def spool_dir(tmp_path: Path) -> Path:
    path = tmp_path / "spool"
    path.mkdir()
    return path


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def transcriber() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def make_app(
    pipeline: FakePipeline, transcriber: FakeTranscriber, spool_dir: Path, settings: Settings
) -> Callable[..., FastAPI]:
    def build(**overrides: object) -> FastAPI:
        options: dict[str, object] = {
            "pipeline": pipeline,
            "transcriber": transcriber,
            "spool_dir": spool_dir,
            "settings": settings,
            "telemetry_setup": lambda: None,
        }
        options.update(overrides)
        return create_app(**options)  # pyright: ignore[reportArgumentType]

    return build


@pytest.fixture
def client(make_app: Callable[..., FastAPI]) -> Iterator[TestClient]:
    with TestClient(make_app()) as test_client:
        yield test_client
