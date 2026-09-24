# T05 — Observability stack

**Lane:** E · **Depends on:** T00

## Goal
Local infra that makes every other lane observable from day one.

## Allowed paths
`infra/`, `src/convintel/telemetry/`, `tests/telemetry/`

## Requirements
- `infra/docker-compose.yml`: Arize Phoenix, Prometheus, Grafana (provisioned datasource +
  dashboards as code), MLflow (Postgres backend + local artifact store), Postgres.
- `convintel.telemetry.setup()`: OpenTelemetry tracer provider exporting OTLP to Phoenix,
  OpenInference instrumentation for LangChain/LangGraph, and a `prometheus_client` registry with **all metrics
  from `docs/ARCHITECTURE.md#telemetry` pre-declared** (other lanes import them, they don't redefine them).
- Helpers: `@traced("operator.intent")` decorator/context manager that records latency into
  `convintel_operator_latency_seconds` and errors into `convintel_operator_errors_total`.
- Grafana dashboard "Pipeline overview": latency p50/p95, error rate, tokens, cost/conversation, verifier rate.

## Acceptance criteria
- [ ] `docker compose up` brings everything up healthy (healthchecks defined).
- [ ] A demo script emits a fake trace that is visible in Phoenix and metrics visible in Grafana.
- [ ] Unit tests for the decorator (latency recorded, errors counted, exception re-raised).
- [ ] `infra/README.md` lists URLs, ports and default credentials (local only).

## Out of scope
Cloud deployment (T13), alert rules (T15).
