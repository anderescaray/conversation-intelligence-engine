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

Warms the four operators concurrently in an `asyncio.TaskGroup`, so startup takes the slowest
one, not their sum: PII loads spaCy/Presidio, sentiment and intent load their models (intent
resolves `intent-classifier@champion` from MLflow), summary only validates the LLM config
(provider wired up, API key present) without a network call. Then it resolves τ and compiles
the graph.

It raises, and the graph stays uncompiled (so `analyze()` keeps refusing to run), when:

- a warmup fails: an `ExceptionGroup` of the warmups that had failed when the others were
  cancelled, each an `OperatorError` carrying `.operator`;
- all warmups together exceed `PIPELINE_WARMUP_TIMEOUT_S` (300 s): `PipelineError("warmup_timeout")`;
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
   instrumentation records node inputs on spans.
3. **Other operators fail open.** An expected failure (`ConvIntelError`) of sentiment, intent
   or summary leaves that field `None` and appends `"<operator>:<ErrorType>"` to `errors`, the
   type only, never the message (e.g. `"summary:OperatorError"`). Anything else is a bug and
   propagates. An LLM result with tokens but no known price adds `"<operator>:UnknownCost"`.
4. **Aggregate.** `total_cost_usd` is the sum of known costs, `total_latency_ms` the
   wall-clock time of `analyze()`, `trace_id` the id of the root span.

## Telemetry

Root span `pipeline.analyze` (`conversation.id`, `pipeline.version`; on failure `error.type`
only); one `operator.<name>` span per operator from `traced()`, and `operator.<name>.warmup`
spans at startup. `convintel_pipeline_latency_seconds{channel}` is observed once per
conversation, failures included. No text or PII in spans, logs or errors.

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

One run makes one summary call, about $0.002, and takes ~15 s (warmups ~3-4 s, `analyze()`
~2-2.5 s, nearly all of it the LLM call).

`setup()` sets the process-wide tracer provider, which OpenTelemetry allows only once per
process. Run the test with `-m integration`, as above: in a single process that also runs
`tests/telemetry/test_setup.py` (e.g. `-m ""`), whichever module calls `setup()` first owns
the provider, and `test_setup.py` would not see its own spans if this test ran first.
