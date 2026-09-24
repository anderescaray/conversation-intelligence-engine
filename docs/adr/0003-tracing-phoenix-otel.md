# ADR-0003: Arize Phoenix over OpenTelemetry for LLM tracing and evals

**Status:** Accepted · **Date:** 2026-09

## Context
We need step-by-step traces of LLM and agent calls, plus eval tooling. Options: LangSmith (hosted),
Langfuse, Arize Phoenix.

## Decision
Arize Phoenix, self-hosted, fed by OpenTelemetry with OpenInference instrumentation.

## Consequences
- OTel-native: the same traces could be sent to any OTel backend (Datadog, Honeycomb, Grafana Tempo),
  with no vendor lock-in and no data sent to third parties (important with conversation data).
- Runs locally in docker compose, at no cost.
- Prometheus remains the source of truth for SLO metrics. Phoenix handles traces and LLM evals.
