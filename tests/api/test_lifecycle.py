import logging
from collections.abc import AsyncIterator, Callable

import pytest
from api_support import SECRET, FakePipeline
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from convintel.api.errors import ServiceNotConfiguredError
from convintel.errors import OperatorError, PipelineError


def requests(route: str, status: str) -> float:
    return (
        REGISTRY.get_sample_value("convintel_requests_total", {"route": route, "status": status})
        or 0.0
    )


def test_healthz_is_live_without_startup(make_app: Callable[..., FastAPI]) -> None:
    response = TestClient(make_app()).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_only_after_warmup(make_app: Callable[..., FastAPI], pipeline: FakePipeline) -> None:
    app = make_app()
    assert TestClient(app).get("/readyz").status_code == 503
    with TestClient(app) as client:
        assert pipeline.warmed
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}
    assert TestClient(app).get("/readyz").status_code == 503


def test_telemetry_is_set_up_before_warmup(make_app: Callable[..., FastAPI]) -> None:
    order: list[str] = []

    class Recording(FakePipeline):
        async def warmup(self) -> None:
            order.append("warmup")

    with TestClient(make_app(pipeline=Recording(), telemetry_setup=lambda: order.append("otel"))):
        pass
    assert order == ["otel", "warmup"]


def test_startup_fails_when_warmup_fails(
    make_app: Callable[..., FastAPI], pipeline: FakePipeline
) -> None:
    pipeline.warmup_error = RuntimeError("model weights missing")
    app = make_app()
    with pytest.raises(RuntimeError), TestClient(app):
        pass
    assert app.state.convintel.ready is False


@pytest.mark.parametrize(
    ("error", "labels"),
    [
        (
            ExceptionGroup(
                "warmups failed",
                [
                    OperatorError("sentiment", f"model missing near {SECRET}"),
                    ExceptionGroup("nested", [OperatorError("intent", SECRET)]),
                    RuntimeError(SECRET),
                ],
            ),
            ["sentiment:OperatorError", "intent:OperatorError", "RuntimeError"],
        ),
        (PipelineError("warmup_timeout"), ["warmup_timeout:PipelineError"]),
        (RuntimeError(SECRET), ["RuntimeError"]),
    ],
    ids=["group", "pipeline-error", "plain"],
)
def test_a_failed_warmup_logs_which_one_failed(
    make_app: Callable[..., FastAPI],
    pipeline: FakePipeline,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    labels: list[str],
) -> None:
    pipeline.warmup_error = error
    app = make_app()

    # Startup still aborts with the original exception, group included.
    with (
        caplog.at_level(logging.ERROR, logger="convintel.api.main"),
        pytest.raises(type(error)) as excinfo,
        TestClient(app),
    ):
        pass

    assert excinfo.value is error
    assert app.state.convintel.ready is False
    records = [r for r in caplog.records if r.name == "convintel.api.main"]
    assert [r.args for r in records] == [(label,) for label in labels]
    # Type names only: messages can quote conversation text.
    assert not any(SECRET in r.getMessage() for r in caplog.records)


def test_startup_fails_without_a_pipeline(make_app: Callable[..., FastAPI]) -> None:
    with pytest.raises(ServiceNotConfiguredError), TestClient(make_app(pipeline=None)):
        pass


def test_metrics_exposes_the_shared_registry(client: TestClient) -> None:
    client.post("/v1/conversations", json={})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "convintel_requests_total" in response.text
    assert "convintel_operator_latency_seconds" in response.text


def test_probes_are_not_counted(client: TestClient) -> None:
    before = [requests(route, "200") for route in ("/healthz", "/readyz", "/metrics")]
    for route in ("/healthz", "/readyz", "/metrics"):
        client.get(route)
    after = [requests(route, "200") for route in ("/healthz", "/readyz", "/metrics")]
    assert after == before


def test_unknown_paths_share_one_label(client: TestClient) -> None:
    before = requests("unmatched", "404")
    client.get("/v1/conversations/some-id-123")
    client.get("/anything/else")
    assert requests("unmatched", "404") == before + 2
    assert requests("/v1/conversations/some-id-123", "404") == 0


def test_failure_after_response_start_counts_as_500_and_propagates(
    make_app: Callable[..., FastAPI],
) -> None:
    class StreamBroke(RuntimeError):
        pass

    async def chunks() -> AsyncIterator[bytes]:
        yield b"partial"
        raise StreamBroke

    async def broken_stream() -> StreamingResponse:
        return StreamingResponse(chunks())

    app = make_app()
    app.add_api_route("/test/broken-stream", broken_stream)
    before = requests("/test/broken-stream", "500")
    with TestClient(app) as client, pytest.raises(StreamBroke):
        client.get("/test/broken-stream")
    assert requests("/test/broken-stream", "500") == before + 1
    assert requests("/test/broken-stream", "200") == 0
