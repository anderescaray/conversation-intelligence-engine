# Sentiment (`convintel.operators.sentiment`)

`SentimentOperator` implements `Operator[SentimentOutput]` on a **redacted** conversation.

- Only customer turns are scored (agent and bot turns are scripted politeness), in one batched
  call to `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- `customer_trajectory`: argmax label per customer turn, in order.
- `overall`: recency-weighted mean polarity `P(pos) - P(neg)`, exponential decay with a
  half-life of 1.5 turns, neutral band ±0.2. Rationale in `trajectory.py`: the final turn alone
  cannot override a consistent history (a closing "ok bye"), but the final two turns, when they
  agree, always do (an angry customer whose problem got fixed ends positive).
- No customer turns → `neutral`, empty trajectory, model not loaded.

## Runtime

- The model is a lock-protected per-process singleton (`get_sentiment_model`).
- `await operator.warmup()` loads the model and runs one inference on a fixed non-PII
  sentence, off the event loop. Call it at startup (`Pipeline.warmup()` does): a missing model
  or dependency fails startup instead of the first request, the first request does not pay
  for loading or CUDA initialization, and `run()` then uses the model that was checked. Span
  `operator.sentiment.warmup` (`operator.version`, `model.id`; on failure `error.type` only,
  no exception event). Failures become
  `OperatorError("sentiment", "warmup failed (<ExceptionType>)")` with no chained cause.
  Without `warmup()`, the model still loads lazily on the first conversation with customer
  turns.
- Inference runs in a worker thread under a process-wide lock (transformers pipelines are not
  thread-safe), so one batch runs at a time per process: **scale with worker processes**, not
  threads.
- Needs `transformers`, `huggingface_hub` and `torch`, all in the `ml` group (ADR-0005),
  imported only when the model loads. The production image must install them (T13).
- The model is **never downloaded at runtime** (AGENTS.md rule 11). It loads only from the
  local Hugging Face cache, at the revision pinned in `PINNED_MODELS` (see Provisioning).
- Errors: any failure becomes
  `OperatorError("sentiment", "sentiment inference failed (<ExceptionType>)")`, with no chained
  cause, so text quoted in third-party messages never reaches spans or logs (same policy as the
  PII redactor).
- `run()` does not open a span: it annotates the caller's current span (a no-op when there is
  none). The pipeline wraps the call with `traced("operator.sentiment")`, which records latency,
  errors and `operator.version` once. Attributes: `conversation.id`, `model.id`,
  `sentiment.customer_turns`, `sentiment.overall`. Never text.

## Provisioning

Fetch the pinned revision once, explicitly, before starting the service (T13 runs this in the
Docker build, with the same `HF_HOME` as the runtime):

```bash
uv run --group ml hf download cardiffnlp/twitter-roberta-base-sentiment-latest config.json merges.txt pytorch_model.bin special_tokens_map.json vocab.json --revision 3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7
```

- The command is `provision_command(DEFAULT_MODEL_ID)`; a unit test keeps this README in sync.
- Revision `3216a57f…` is the commit `refs/main` pointed to in the local HF cache on
  2026-11-07 (read from the cache, no network call). The file list is exactly what loading
  needs, so the download is the set that was tested.
- If the snapshot or any of its files is missing, `warmup()` raises
  `OperatorError("sentiment", "model <id>@<revision> is missing or incomplete in the local
  Hugging Face cache; provision it with: <command>")` and the service does not start. A model
  id without a pinned revision is refused too; `main` is never resolved.
- The snapshot is resolved with `snapshot_download(..., local_files_only=True)` (a cache
  lookup) and its *directory* is passed to `transformers.pipeline`. Passing the repo id is not
  enough even with `local_files_only=True`: for `.bin` weights transformers starts a
  safetensors auto-conversion thread that calls the hub.

## Tests

- Unit: `FakeSentimentModel` (no transformers, no torch, no network).
- Integration (`-m integration`, skipped without the `ml` group): real model on a synthetic
  call, and a warmup with every socket connection refused, to prove startup needs no network.
  Needs the provisioned cache.
