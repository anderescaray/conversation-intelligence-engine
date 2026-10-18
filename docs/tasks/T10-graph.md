# T10 — LangGraph pipeline (orchestration)

Owner: Ander (human). Branch: `lane-g/T10-graph`.
Issue follow-ups addressed: #20 (threshold policy, `unknown`, tau override).

## Goal
One entry point, `Pipeline.analyze(conversation) -> ConversationInsights`, that runs every
operator in the right order, in parallel where possible, calls the LLM intent verifier only
when the classifier is unsure, and returns partial results when a non-critical operator fails.
T12 (API) only builds the pipeline at startup and calls `analyze()`.

## Allowed paths
- `src/convintel/pipeline/` (new): `state.py`, `graph.py`, `verifier.py`, `__init__.py`
- `prompts/intent_verifier/v1.yaml` (new)
- `tests/pipeline/` (new)
- `src/convintel/config.py` + `.env.example`: one new field (human-owned, done by Ander)
Contracts are frozen. Operators are not modified (follow-ups go to issues).

## Graph
    pii ──► sentiment ─────────────────┐
        ├─► intent ──► [verify?] ──────┤──► aggregate
        │              └─► verifier ───┤
        └─► summary ───────────────────┘
- State: the input `Conversation`, the redacted conversation, PII entity count, one
  `OperatorResult | None` per operator, the list of errors and the effective tau.
- Fan-out after `pii`: sentiment, intent and summary run in parallel on the REDACTED text.
- Conditional edge after `intent` -> `verifier` only if ALL of:
  label != "unknown", and confidence < tau.
  The critical path is max(summary, intent + verifier).

## Decisions
1. **PII is fail-closed.** If redaction fails, the pipeline aborts with `PipelineError`
   (T12 -> 5xx). No model may ever see unredacted text.
2. **Other operators fail open.** A failed sentiment/intent/summary sets that field to `None`
   and appends `"<operator>:<ErrorType>"` to `errors` (type only, never the message).
3. **Tau policy.** Effective tau = `settings.intent_verifier_threshold` if it is not None, else
   the model bundle's threshold. If both are None, `warmup()` raises and the service does not
   start (fail fast at startup, never per request). The effective tau is recorded on the span.
4. **No verifier for `unknown`** (no customer turns): nothing to classify.
5. **Verifier** (`prompts/intent_verifier/v1.yaml`, via `convintel.llm.client`): the LLM picks
   one label from the classifier's top-k candidates (structured output constrained to those
   labels). The result keeps `confidence` = the classifier's calibrated probability for the
   chosen label, sets `source="llm_verifier"`, and adds the verifier's latency, tokens and cost
   to the intent result.
6. **Verifier failure** -> keep the classifier result unchanged and record the error;
   the conversation still gets an intent.
7. **Totals.** `total_cost_usd` = sum of known costs; `total_latency_ms` = wall-clock time of
   `analyze()`; `trace_id` from the current OTel span; `pipeline_version` is a constant here.

## Telemetry
- Root span `pipeline.analyze`; operator spans come from `@traced` (T05).
- `convintel_pipeline_latency_seconds{channel}` observed once per conversation.
- `convintel_intent_verifier_invocations_total{outcome}`: `confirmed` | `overridden` | `failed`.
  Verifier rate = invocations / pipeline count.
- No text or PII in spans, logs or errors.

## Acceptance criteria
- Unit tests with fake operators (no models, no network): happy path; low confidence ->
  verifier; high confidence -> no verifier; `unknown` -> no verifier; tau override vs bundle;
  no tau -> `warmup()` raises; PII failure aborts; summary failure -> partial result;
  verifier failure -> classifier result kept; operators run in parallel (timing or call order).
- One `@pytest.mark.integration` test with the real operators and Ollama.
- ruff, ruff format --check, pyright, pytest, uv lock --check.

## Out of scope
API and Postgres (T12), tau re-tuning on the golden set (T11), the T16 agent.
