# ADR-0004: Prompts as versioned, immutable files

**Status:** Accepted · **Date:** 2026-09

## Context
Prompt changes alter behaviour as much as model changes, but are often made ad hoc and untracked.

## Decision
- Prompts live in `prompts/<name>/v<N>.yaml` and are immutable once merged. Changes add a new version.
- Each `OperatorResult` records `prompt_version`, and traces carry it as an attribute.
- Prompts are optionally mirrored to the MLflow prompt registry, so models and prompts share one lineage view.
- A prompt version is promoted only if the eval gate (T11) passes against the current champion.

## Consequences
Any production result can be traced back to the exact prompt, model and pipeline version.
Canary releases (T14) can compare prompt versions on live traffic.
