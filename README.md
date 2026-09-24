# Conversation Intelligence Engine

> Extracting meaning from voice and messaging conversations: production-grade, observable, and cost-aware.

Ingests calls and chats (Twilio webhooks or batch), transcribes them, redacts PII, and runs
sentiment, intent and summarisation operators orchestrated with **LangGraph**. A fine-tuned
classifier handles the easy cases and an LLM verifies the uncertain ones. Every step is traced
(**OpenTelemetry → Arize Phoenix**), measured against SLOs (**Prometheus/Grafana**), versioned
(**MLflow** models and prompts) and gated by offline evals in CI.

<!-- TODO: demo GIF / Loom link -->

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full diagram, telemetry and SLOs.

## Results

<!-- Fill in after T03 / T10 / T11 / T15 -->

| Intent strategy | Macro-F1 | p95 latency | Cost / 1k conversations |
|---|---|---|---|
| Classifier only | – | – | – |
| LLM only | – | – | – |
| **Hybrid (classifier + LLM verifier)** | – | – | – |

| SLO | Target | Measured |
|---|---|---|
| Text p95 end-to-end latency | < 4 s | – |
| Voice real-time factor p95 | < 0.5 | – |
| Error rate | < 1 % | – |

## Quickstart

```bash
uv sync
docker compose -f infra/docker-compose.yml up -d
uv run uvicorn convintel.api.main:app --reload
```

## Key decisions

- [ADR-0001 Small model first, LLM as verifier](docs/adr/0001-small-model-first.md)
- [ADR-0002 LangGraph for orchestration](docs/adr/0002-langgraph-orchestration.md)
- [ADR-0003 Phoenix over OpenTelemetry](docs/adr/0003-tracing-phoenix-otel.md)
- [ADR-0004 Versioned prompts](docs/adr/0004-prompt-versioning.md)

## How I built this

Built as tech lead of a team of coding agents (Claude Code, Pi, Codex) running in parallel git
worktrees via Orca, with frozen contracts, per-task briefs and human review on every PR.
See [`docs/WORKFLOW.md`](docs/WORKFLOW.md) and the engineering log in [`docs/LOG.md`](docs/LOG.md).

<!-- TODO: what worked, what didn't, agent bake-off results -->
