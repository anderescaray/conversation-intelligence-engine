import socket
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from convintel.config import get_settings
from convintel.operators.intent import model as model_module

# `training/` is a top-level directory, not part of the installed package; its pure helpers
# (split, config, calibration study) are unit-tested from here.
_ROOT = str(Path(__file__).resolve().parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("integration"):
        return

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unit tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture(autouse=True)
def isolated_model_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No cached model or settings leak between tests, and no local .env into assertions."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    get_settings.cache_clear()
    model_module._cached_model.cache_clear()  # pyright: ignore[reportPrivateUsage]
    yield
    get_settings.cache_clear()
    model_module._cached_model.cache_clear()  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    yield exporter
    provider.shutdown()
