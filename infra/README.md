# Local observability stack

Everything runs via `docker compose` from this directory (or with `-f` from the repo root).
All credentials below are **local-only** — nothing here is deployed anywhere.

```bash
docker compose -f infra/docker-compose.yml up -d     # start everything
docker compose -f infra/docker-compose.yml ps        # check health
docker compose -f infra/docker-compose.yml down      # stop (add -v to wipe data)
```

## Services

| Service    | URL / port                          | Credentials (local only)                    |
|------------|-------------------------------------|---------------------------------------------|
| Phoenix    | http://localhost:6006 (UI + OTLP/HTTP) | —                                        |
| Phoenix    | `localhost:4317` (OTLP/gRPC)         | —                                           |
| Prometheus | http://localhost:9090                | —                                           |
| Grafana    | http://localhost:3000                | `admin` / `admin`                           |
| MLflow     | http://localhost:5000                | —                                           |
| Postgres   | `localhost:5432`                     | `convintel` / `convintel`, DBs `convintel` + `mlflow` |

## How the pieces connect

- **Phoenix** receives traces over OTLP/HTTP at `http://localhost:6006/v1/traces`
  (`PHOENIX_COLLECTOR_ENDPOINT` in `.env`). `convintel.telemetry.setup()` points there.
- **Prometheus** scrapes:
  - itself (`localhost:9090`),
  - `host.docker.internal:8000` — the FastAPI service (`/metrics`, T12),
  - `host.docker.internal:9100` — the demo script below.
- **Grafana** is provisioned as code: the Prometheus datasource lives in
  `grafana/provisioning/datasources/` and dashboards (e.g. *Pipeline overview*) in
  `grafana/dashboards/`. Changes to those files are picked up within ~10 s.
- **MLflow** uses Postgres (`mlflow` database) as backend store and a named volume
  (`mlflow-artifacts`) as artifact root. Track with `MLFLOW_TRACKING_URI=http://localhost:5000`.

## Demo: trace in Phoenix + metrics in Grafana

With the stack healthy, run on the host:

```bash
uv run python -m convintel.telemetry.demo
```

It emits a synthetic `conversation.process` trace (child spans `operator.*` and
`llm.call`) every 10 s and serves metrics on `:9100` for Prometheus to scrape.
Then:

- Phoenix → project `convintel` → see the trace tree.
- Grafana → *Pipeline overview* → latency p50/p95, error rate, verifier rate.
- Prometheus → query e.g. `convintel_operator_latency_seconds_count`.

Ctrl+C stops the demo; the stack keeps running.

## Hiding LLM inputs/outputs in traces

OpenInference captures LLM request/response payloads on spans by default — that
is what makes local debugging in Phoenix useful. It is safe here by design:
text only reaches an LLM call **after** the PII redaction boundary (Presidio
runs before the pipeline), so spans never carry raw PII.

If you ever need to hide payloads anyway (e.g. demoing Phoenix externally), set
these env vars on the app process before `convintel.telemetry.setup()` runs:

```bash
OPENINFERENCE_HIDE_INPUTS=true          # drop input.value / input.mime_type
OPENINFERENCE_HIDE_OUTPUTS=true         # drop output.value / output.mime_type
OPENINFERENCE_HIDE_INPUT_MESSAGES=true  # drop llm.input_messages
OPENINFERENCE_HIDE_OUTPUT_MESSAGES=true # drop llm.output_messages
OPENINFERENCE_HIDE_INPUT_TEXT=true      # keep structure, drop message text
OPENINFERENCE_HIDE_OUTPUT_TEXT=true
```

The full list lives in `openinference.instrumentation.config`. They are plain
environment variables, not `Settings` fields — wiring them into
`convintel.config` is a human-owned follow-up (config is frozen for lanes).

## Data persistence

Named volumes (`postgres-data`, `phoenix-data`, `grafana-data`,
`mlflow-artifacts`) survive `down`. Use `down -v` to reset everything.
