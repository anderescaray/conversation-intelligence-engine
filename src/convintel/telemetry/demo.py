"""Emit a fake conversation trace to Phoenix and expose metrics for Prometheus to scrape.

Run with the observability stack up (`docker compose -f infra/docker-compose.yml up -d`):

    uv run python -m convintel.telemetry.demo

Then open Phoenix (http://localhost:6006) for the `conversation.process` trace and
Grafana (http://localhost:3000, "Pipeline overview") for the metrics. Ctrl+C to stop.

All conversation content is synthetic; nothing here contains real PII.
"""

import random
import time
import uuid

from opentelemetry import trace
from prometheus_client import start_http_server

from convintel.telemetry import (
    ASR_REALTIME_FACTOR,
    INTENT_VERIFIER_INVOCATIONS_TOTAL,
    PIPELINE_LATENCY_SECONDS,
    REQUESTS_TOTAL,
    setup,
    traced,
)

_METRICS_PORT = 9100
_EMIT_INTERVAL_S = 10.0


def _fake_operator(name: str, *, fail: bool = False) -> None:
    @traced(f"operator.{name}", version="0.1.0-demo")
    def _run() -> None:
        time.sleep(random.uniform(0.05, 0.3))
        if fail:
            raise RuntimeError(f"synthetic {name} failure")

    _run()


def _emit_one(tracer: trace.Tracer) -> None:
    conversation_id = f"demo-{uuid.uuid4().hex[:8]}"
    channel = random.choice(["voice", "chat"])
    start = time.perf_counter()

    with tracer.start_as_current_span("conversation.process") as root:
        root.set_attribute("conversation.id", conversation_id)
        root.set_attribute("channel", channel)

        if channel == "voice":
            ASR_REALTIME_FACTOR.labels(model="faster-whisper-small").observe(
                random.uniform(0.2, 0.6)
            )

        _fake_operator("sentiment")
        _fake_operator("intent")

        if random.random() < 0.4:
            INTENT_VERIFIER_INVOCATIONS_TOTAL.labels(
                outcome=random.choice(["agreed", "overrode"])
            ).inc()

        _fake_operator("summary", fail=random.random() < 0.25)

        with tracer.start_as_current_span("llm.call") as llm_span:
            llm_span.set_attribute("model.id", "demo-model")
            llm_span.set_attribute("llm.tokens.input", 512)
            llm_span.set_attribute("llm.tokens.output", 128)
            llm_span.set_attribute("llm.cost_usd", 0.0008)
            time.sleep(random.uniform(0.1, 0.4))

    PIPELINE_LATENCY_SECONDS.labels(channel=channel).observe(time.perf_counter() - start)
    REQUESTS_TOTAL.labels(route="/v1/conversations", status="200").inc()


def main() -> None:
    setup()
    start_http_server(_METRICS_PORT)
    tracer = trace.get_tracer("convintel.telemetry.demo")
    print(
        f"Emitting fake traces and metrics every {_EMIT_INTERVAL_S:.0f}s.\n"
        f"  Phoenix:    http://localhost:6006  (trace: conversation.process)\n"
        f"  Prometheus: http://localhost:9090  (query: convintel_operator_latency_seconds)\n"
        f"  Grafana:    http://localhost:3000  (dashboard: Pipeline overview)\n"
        f"Metrics scraped from 0.0.0.0:{_METRICS_PORT}. Ctrl+C to stop."
    )
    try:
        while True:
            _emit_one(tracer)
            time.sleep(_EMIT_INTERVAL_S)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
