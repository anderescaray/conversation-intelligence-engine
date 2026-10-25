import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from api_support import NOW, SECRET, FakePipeline, FakeTranscriber, write_wav
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY, generate_latest

from convintel.errors import (
    ConvIntelError,
    LLMError,
    OperatorError,
    TranscriptionError,
)

ROUTE = "/v1/conversations"


def chat(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "conv-001",
        "channel": "chat",
        "started_at": NOW.isoformat(),
        "turns": [{"index": 0, "speaker": "customer", "text": f"Card blocked. {SECRET}"}],
    }
    body.update(overrides)
    return body


def requests(status: str) -> float:
    labels = {"route": ROUTE, "status": status}
    return REGISTRY.get_sample_value("convintel_requests_total", labels) or 0.0


def test_returns_insights(client: TestClient, pipeline: FakePipeline) -> None:
    before = requests("200")
    response = client.post(ROUTE, json=chat())
    assert response.status_code == 200
    assert response.json()["conversation_id"] == "conv-001"
    assert response.json()["pipeline_version"] == "fake-1"
    assert pipeline.seen[0].turns[0].text.startswith("Card blocked.")
    assert requests("200") == before + 1


def test_span_carries_ids_only(client: TestClient, spans: InMemorySpanExporter) -> None:
    client.post(ROUTE, json=chat())
    (span,) = [s for s in spans.get_finished_spans() if s.name == "api.conversations"]
    assert dict(span.attributes or {}) == {"conversation.id": "conv-001", "channel": "chat"}


def test_rejects_conversations_without_content(client: TestClient, pipeline: FakePipeline) -> None:
    response = client.post(ROUTE, json=chat(turns=[]))
    assert response.status_code == 422
    assert not pipeline.seen


def test_validation_errors_do_not_echo_input(client: TestClient) -> None:
    bad = chat(turns=[{"index": -1, "speaker": "narrator", "text": SECRET}])
    response = client.post(ROUTE, json=bad)
    assert response.status_code == 422
    assert SECRET not in response.text
    assert response.json()["detail"] == "Invalid request body."
    assert {e["type"] for e in response.json()["errors"]} >= {"enum", "greater_than_equal"}


def test_transcribes_spooled_audio(
    client: TestClient, pipeline: FakePipeline, transcriber: FakeTranscriber, spool_dir: Path
) -> None:
    audio = write_wav(spool_dir / "call.wav")
    body = chat(channel="voice", turns=[], audio_uri="call.wav", metadata={"source": "batch"})
    response = client.post(ROUTE, json=body)
    assert response.status_code == 200
    assert transcriber.calls == [(str(audio.resolve()), "conv-001")]
    analyzed = pipeline.seen[0]
    assert analyzed.turns[0].text == "synthetic transcript"
    assert analyzed.started_at == NOW
    assert analyzed.metadata == {"source": "batch", "asr_model": "fake"}


@pytest.mark.parametrize("turns", [[], None])
def test_audio_outside_the_spool_never_reaches_the_pipeline(
    client: TestClient,
    pipeline: FakePipeline,
    transcriber: FakeTranscriber,
    tmp_path: Path,
    turns: list[object] | None,
) -> None:
    outside = write_wav(tmp_path / "elsewhere.wav")
    body = chat(audio_uri=str(outside)) if turns is None else chat(turns=turns, audio_uri="../x")
    response = client.post(ROUTE, json=body)
    assert response.status_code == 422
    assert response.json() == {"detail": "Audio rejected.", "reason": "outside_spool"}
    assert not pipeline.seen
    assert not transcriber.calls


def test_overlong_audio_is_413(make_app: Callable[..., FastAPI], spool_dir: Path) -> None:
    write_wav(spool_dir / "long.wav", seconds=2)
    app = make_app()
    app.state.convintel.spool.max_duration_s = 1
    with TestClient(app) as client:
        response = client.post(ROUTE, json=chat(turns=[], audio_uri="long.wav"))
    assert response.status_code == 413
    assert response.json()["reason"] == "too_long"


@pytest.mark.parametrize(
    ("error", "status", "message"),
    [
        (OperatorError("sentiment", f"failed on {SECRET}"), 500, "Conversation analysis failed."),
        (LLMError(f"provider said {SECRET}"), 502, "An upstream model call failed."),
        (TranscriptionError(SECRET), 500, "Transcription failed."),
        (ConvIntelError(SECRET), 500, "The request could not be processed."),
        (RuntimeError(SECRET), 500, "Internal server error."),
    ],
)
def test_failures_return_fixed_messages_and_leak_nothing(
    make_app: Callable[..., FastAPI],
    pipeline: FakePipeline,
    spans: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    status: int,
    message: str,
) -> None:
    pipeline.error = error
    before = requests(str(status))
    caplog.set_level(logging.DEBUG)
    with TestClient(make_app()) as client:
        response = client.post(ROUTE, json=chat())
    assert response.status_code == status
    assert response.json() == {"detail": message}
    assert requests(str(status)) == before + 1

    assert SECRET not in response.text
    for record in caplog.records:
        assert SECRET not in record.getMessage()
        assert record.exc_info is None
    for span in spans.get_finished_spans():
        assert SECRET not in str(dict(span.attributes or {}))
        assert not span.events
    assert SECRET not in generate_latest(REGISTRY).decode()
