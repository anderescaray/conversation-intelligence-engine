# Pipeline (`convintel.pipeline`)

One entry point for the API: build the pipeline at startup, warm it, then call `analyze()` per
conversation.

```python
from convintel.config import Settings
from convintel.pipeline.factory import build_pipeline

pipeline = build_pipeline(Settings())  # cheap, no I/O
await pipeline.warmup()  # loads every model; raising aborts startup
insights = await pipeline.analyze(conversation)  # -> ConversationInsights
```

## `warmup()`

Warms the four operators and the intent verifier concurrently in an `asyncio.TaskGroup`, so
startup takes the slowest one, not their sum: PII loads spaCy/Presidio, sentiment and intent
load their models (intent resolves `intent-classifier@champion` from MLflow), summary only
validates the LLM config (provider wired up, API key present) without a network call, and the
LLM verifier, when enabled, loads its prompt and validates the same config. Then it resolves τ
and compiles the graph.

It raises, and the graph stays uncompiled (so `analyze()` keeps refusing to run), when:

- a warmup fails: an `ExceptionGroup` of the warmups that had failed when the others were
  cancelled, each an `OperatorError` carrying `.operator` (`intent_verifier` for the verifier);
- all warmups together exceed the timeout: `PipelineError("warmup_timeout")`. `Pipeline` takes
  it as the required `warmup_timeout_s`; `build_pipeline` passes the `PIPELINE_WARMUP_TIMEOUT_S`
  setting (300 s by default; keep it below the platform's startup probe);
- neither `INTENT_VERIFIER_THRESHOLD` nor the model bundle provides τ:
  `PipelineError("intent_threshold")`.

## `analyze()`

1. **PII redaction, fail closed.** Always runs, whatever the input's `redacted` flag says. If
   the redactor fails, or returns a conversation not flagged as redacted, `analyze()` raises
   `PipelineError("pii")` with no chained cause (the original error may quote the text it
   failed to redact). No model ever sees unredacted text.
2. **Graph on the redacted conversation.** Sentiment, intent and summary run in parallel;
   intent goes to the verifier only for a real label with confidence strictly below τ. The
   graph state only ever holds the redacted conversation, because the LangChain
   instrumentation records node inputs on spans. See [Intent verifier](#intent-verifier).
3. **Other operators fail open.** An expected failure (`ConvIntelError`) of sentiment, intent
   or summary leaves that field `None` and appends `"<operator>:<ErrorType>"` to `errors`, the
   type only, never the message (e.g. `"summary:OperatorError"`). Anything else is a bug and
   propagates. An LLM result with tokens but no known price adds `"<operator>:UnknownCost"`.
4. **Aggregate.** `total_cost_usd` is the sum of known costs, `total_latency_ms` the
   wall-clock time of `analyze()`, `trace_id` the id of the root span.

## Intent verifier

A second opinion on a low-confidence intent, in `verifier.py`. `build_pipeline` plugs in
`LLMIntentVerifier` only with `INTENT_VERIFIER_ENABLED=true`; the default is
`KeepClassifierLabel`, which keeps the classifier's label. The flag stays off until the verifier
passes its evaluation gate.

- **Candidates.** The classifier's top-5 (see `training/intent/reports/verifier-candidates.md`).
  `build_pipeline` builds `IntentOperator(top_k=5)` only when the verifier is enabled; with the
  flag off the classifier keeps its default `top_k`, so the output is unchanged. Candidates
  are shown to the LLM as letters with their taxonomy-v3 descriptions
  (`intent_descriptions(3)`), never the label names, the probabilities or which candidate the
  classifier preferred. Their order is shuffled with a
  seed from `sha256(conversation.id)`: the same conversation always gets the same prompt, and
  the classifier's top-1 lands on each letter about equally often. Fewer candidates show
  fewer letters.
- **Input.** The full redacted conversation with speakers, wrapped like the summary's; the
  prompt (`prompts/intent_verifier/v1.yaml`) classifies the customer's reason for contacting,
  with agent and bot turns as context only.
- **Answer.** `answer_model(n)`: `reasoning` (one or two sentences) first, then `choice`, one
  of the first `n` letters or `NONE`. A letter with no candidate fails validation, gets the
  client's single repair, then `LLMError`. Our code drops `reasoning` on parse and never emits
  it (not on our spans, logs, errors or the insights); see [Telemetry](#telemetry) for what
  OpenInference records.
- **Outcomes.** `confirmed` (the classifier's top-1), `overridden` (another candidate, with the
  classifier's calibrated probability for it), `rejected` (`NONE`: label `unknown`,
  confidence 0, `top_k` kept). All three set `source="llm_verifier"`, keep `operator="intent"`,
  carry the verifier's model and prompt, and sum the classifier's and the verifier's latency,
  tokens and cost; as in `total_cost_usd`, an unpriced LLM call makes the cost unknown (None),
  never zero. `failed` (a `ConvIntelError`, in practice `LLMError`) keeps the classifier's
  result and adds `"intent_verifier:LLMError"` to `errors`; any other exception propagates.

## Telemetry

Root span `pipeline.analyze` (`conversation.id`, `pipeline.version`, `intent.threshold` — the
effective τ, override or bundle; on failure `error.type` only); one `operator.<name>` span per
operator from `traced()`, and `operator.<name>.warmup` spans at startup. The verifier runs in
`operator.intent_verifier` (its `operator.version` is `placeholder` while disabled) with
`verifier.outcome`, `verifier.candidates`, `prompt.version`, `model.id`, `llm.tokens.*` and
`llm.cost_usd`, and increments
`convintel_intent_verifier_invocations_total{outcome=confirmed|overridden|rejected|failed}`.
`convintel_pipeline_latency_seconds{channel}` is observed once per conversation, failures
included.

Our code puts no conversation text, `reasoning` or PII in spans, logs or errors. OpenInference's
LangChain auto-instrumentation (`telemetry.setup()`) does record every LLM call's inputs and
outputs on its own spans: for the summary and the verifier that is the redacted conversation,
the model's answer (the verifier's `reasoning` included) and, on a repair, the client's repair
message, which quotes the previous answer. This is a known exposure of redacted text only (PII
never reaches the graph) until `OPENINFERENCE_HIDE_*` is enabled in production, the planned T13
follow-up.

## Tests

Unit tests (`tests/pipeline/`, fake operators, no models or network) run with `uv run pytest`.

The end-to-end test, `tests/pipeline/test_pipeline_integration.py`, runs the production
pipeline on a synthetic conversation with planted PII and the production telemetry
(`telemetry.setup()` with an in-memory exporter). It checks the insights and that no planted
value reaches any span. It needs:

- the stack up: `docker compose -f infra/docker-compose.yml up -d`, with MLflow serving
  `intent-classifier@champion` (see `training/intent/README.md`);
- the `ml` group: `uv sync --group ml` (it skips without it);
- `ANTHROPIC_API_KEY` in the environment or `.env` (the summary uses the `Settings`
  provider and model, `claude-haiku-4-5` by default).

```bash
uv run pytest -m integration tests/pipeline/test_pipeline_integration.py
```

With the verifier disabled (the default, `INTENT_VERIFIER_ENABLED=false`), one run makes one
summary call, about $0.002, and takes ~15 s (warmups ~3-4 s, `analyze()` ~2-2.5 s, nearly all
of it the LLM call).

`setup()` sets the process-wide tracer provider, which OpenTelemetry allows only once per
process. Run the test with `-m integration`, as above: in a single process that also runs
`tests/telemetry/test_setup.py` (e.g. `-m ""`), whichever module calls `setup()` first owns
the provider, and `test_setup.py` would not see its own spans if this test ran first.
