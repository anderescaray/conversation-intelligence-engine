"""Prometheus metrics for the whole pipeline, pre-declared so lanes import, never redefine.

Metric names and labels are fixed by `docs/ARCHITECTURE.md#telemetry`; changing one breaks
the provisioned Grafana dashboards and every other lane's instrumentation.

`convintel_llm_tokens_total` and `convintel_llm_cost_usd_total` are intentionally absent:
they live in `convintel.llm.metrics` (T04), next to the client that emits them.
"""

from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "convintel_requests_total",
    "HTTP requests served by the API.",
    labelnames=("route", "status"),
)

PIPELINE_LATENCY_SECONDS = Histogram(
    "convintel_pipeline_latency_seconds",
    "End-to-end latency of one conversation through the pipeline.",
    labelnames=("channel",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
)

OPERATOR_LATENCY_SECONDS = Histogram(
    "convintel_operator_latency_seconds",
    "Latency of a single operator invocation.",
    labelnames=("operator", "version"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
)

OPERATOR_ERRORS_TOTAL = Counter(
    "convintel_operator_errors_total",
    "Operator invocations that raised.",
    labelnames=("operator", "error_type"),
)

INTENT_VERIFIER_INVOCATIONS_TOTAL = Counter(
    "convintel_intent_verifier_invocations_total",
    "Times the LLM verifier was invoked on a low-confidence intent.",
    labelnames=("outcome",),
)

ASR_REALTIME_FACTOR = Histogram(
    "convintel_asr_realtime_factor",
    "ASR processing time divided by audio duration (lower is better).",
    labelnames=("model",),
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
)
