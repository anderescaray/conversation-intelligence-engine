import socket
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from convintel.config import get_settings


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--asr-model-path", help="Preinstalled faster-whisper small model directory")


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic stereo.wav"
    time = np.arange(16000 * 10) / 16000
    left = np.sin(time * 2 * np.pi * 220) * 12000
    right = np.sin(time * 2 * np.pi * 440) * 6000
    pcm = np.column_stack((left, right)).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm.tobytes())
    return path


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    path = tmp_path / "model"
    path.mkdir()
    for name in (
        "model.bin",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "vocabulary.txt",
    ):
        (path / name).write_text("synthetic model placeholder", encoding="utf-8")
    return path


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    yield exporter
    provider.shutdown()
