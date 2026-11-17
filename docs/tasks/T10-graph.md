# T10 — LangGraph pipeline (orchestration)

Owner: Ander (human). Branch: `lane-g/T10-graph`, split into the stacked branches
`lane-g/T10-1-…` to `lane-g/T10-5-…`.
Issue follow-ups addressed: #20 (threshold policy, `unknown`, tau override).

## Goal
One entry point, `Pipeline.analyze(conversation) -> ConversationInsights`, that redacts PII
first, then runs the other operators in parallel on the redacted text, routes a low-confidence
intent to a verifier hook, and returns partial results when a non-critical operator fails.
T12 (API) only builds the pipeline at startup with `build_pipeline(settings)`, awaits
`warmup()`, and calls `analyze()`.

## Allowed paths
- `src/convintel/pipeline/` (new): `state.py`, `nodes.py`, `graph.py`, `pipeline.py`,
  `factory.py`, `__init__.py`, `README.md`
- `tests/pipeline/` (new)
- `src/convintel/config.py` + `.env.example`: one new field (human-owned, done by Ander)
- `src/convintel/errors.py`: `PipelineError`; `pyproject.toml` + `uv.lock`: `langgraph` as a
  direct dependency
- Cross-lane changes approved by the human: the intent, PII, sentiment and summary operators
  annotate the caller's span instead of opening their own; PII, sentiment and summary get a
  `warmup()`; `LLMClient` gets a read-only `settings` property.
Contracts are frozen.

## Graph
    analyze():  pii (outside the graph, fail closed)
                  │ redacted conversation
                  ▼
    graph:  START ─► sentiment ────────────────────────► END
                  ├─► summary ──────────────────────────► END
                  └─► intent ──► [verify?] ─────────────► END
                                  └─► verify_intent ─────► END
- PII redaction runs in `analyze()`, outside the graph, on every call, whatever the input's
  `redacted` flag says. The graph only ever holds the redacted conversation, because the
  LangChain instrumentation records node inputs on spans.
- State: the redacted conversation, one `OperatorResult | None` per graph operator
  (sentiment, intent, summary) and the list of errors.
- Sentiment, intent and summary run in parallel on the REDACTED text.
- There is no aggregate node: the graph ends when every branch has ended, and `analyze()`
  builds the insights from the final state.
- Conditional edge after `intent` -> `verify_intent` only if ALL of:
  label != "unknown", and confidence < tau.
  The critical path is max(summary, intent + verifier).

## Decisions
1. **PII is fail-closed.** If redaction fails, or the redactor returns a conversation not
   flagged as redacted, the pipeline aborts with `PipelineError` (T12 -> 5xx). No model may
   ever see unredacted text.
2. **Other operators fail open, only for `ConvIntelError`.** An expected failure of
   sentiment/intent/summary sets that field to `None` and appends `"<operator>:<ErrorType>"`
   to `errors` (type only, never the message). Any other exception is a bug and propagates.
3. **Tau policy.** Effective tau = `settings.intent_verifier_threshold` if it is not None, else
   the model bundle's threshold. If both are None, `warmup()` raises and the service does not
   start (fail fast at startup, never per request). The effective tau is recorded on the span.
4. **No verifier for `unknown`** (no customer turns): nothing to classify.
5. **Verifier hook.** The graph takes the verifier as an injected function; a failed
   verification keeps the classifier result unchanged and records the error. In this stack
   the pipeline's default verifier is a placeholder that keeps the classifier's label; the
   LLM verifier is out of scope (see below).
6. **Startup.** `build_pipeline(settings)` builds the real operators and does no I/O; settings
   are passed in explicitly. `warmup()` warms PII, sentiment, summary and intent concurrently
   in an `asyncio.TaskGroup`, bounded by a timeout; then it resolves tau and compiles the
   graph. A failed warmup, the timeout, or a missing tau aborts startup and leaves the graph
   uncompiled, so `analyze()` keeps refusing to run.
7. **Totals.** `total_cost_usd` = sum of known costs; `total_latency_ms` = wall-clock time of
   `analyze()`; `trace_id` from the current OTel span; `pipeline_version` is a constant here.

## Telemetry
- Root span `pipeline.analyze`. Operators do not instrument themselves: they annotate the
  current span, and the pipeline wraps every operator call with `traced()` (T05), which
  emits the `operator.<name>` span and the operator latency/error metrics.
- `convintel_pipeline_latency_seconds{channel}` observed once per conversation.
- No text or PII in spans, logs or errors.

## Acceptance criteria
- Unit tests with fake operators (no models, no network): happy path; low confidence ->
  verifier; high confidence -> no verifier; `unknown` -> no verifier; tau override vs bundle;
  no tau -> `warmup()` raises; PII failure aborts; summary failure -> partial result;
  verifier failure -> classifier result kept; operators run in parallel (timing or call order);
  warmups run concurrently, a failed warmup or the timeout aborts startup.
- One `@pytest.mark.integration` test with the real operators. The summary runs on the
  production path: Anthropic `claude-haiku-4-5` (the `Settings` defaults), not Ollama.
- ruff, ruff format --check, pyright, pytest, uv lock --check.

### Status (verifier PR, closes #41)
- [x] Effective τ recorded on a span (decision 3): `intent.threshold` on `pipeline.analyze`,
  `test_the_effective_threshold_is_recorded_on_the_root_span`.
- [x] Operators run in parallel inside the graph: `asyncio.Barrier(3)` that sentiment, summary
  and intent must all reach, `test_sentiment_summary_and_intent_run_in_parallel`.

## Out of scope
- The LLM intent verifier. It will be its own PR: its candidates are the classifier's top-5;
  "none of these" -> `unknown` with `outcome="rejected"`; it is enabled only after an
  evaluation gate.
- API and Postgres (T12), tau re-tuning on the golden set (T11), the T16 agent.
