# ADR-0001: Small model first, LLM as verifier

**Status:** Accepted · **Date:** 2026-09

## Context
Intent classification runs on every conversation. At Twilio scale, sending every conversation to an LLM
is slow and expensive, and its accuracy is not clearly better on a closed label set.

## Decision
Use a fine-tuned, calibrated DistilBERT as the primary intent classifier. When its calibrated
confidence is below τ, the LangGraph pipeline routes to an LLM verifier, which sees the top-k candidates
and the conversation and either confirms or overrides the label.

## Consequences
- Most traffic is served in milliseconds at near-zero marginal cost.
- LLM spend concentrates on the hard cases. The verifier rate is a tracked metric.
- Requires calibration and a threshold study (T03), plus a comparison table: classifier only,
  LLM only, and hybrid, on accuracy, p95 latency and cost per 1k conversations. That table goes in the README.
