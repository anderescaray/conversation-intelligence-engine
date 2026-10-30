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
- Needs `transformers` and `torch`, both in the `ml` group (ADR-0005), imported only when the
  model loads. The production image must install them (T13). The model weights are fetched
  from the Hugging Face hub on first load and cached.
- Errors: any failure becomes
  `OperatorError("sentiment", "sentiment inference failed (<ExceptionType>)")`, with no chained
  cause, so text quoted in third-party messages never reaches spans or logs (same policy as the
  PII redactor).
- `run()` does not open a span: it annotates the caller's current span (a no-op when there is
  none). The pipeline wraps the call with `traced("operator.sentiment")`, which records latency,
  errors and `operator.version` once. Attributes: `conversation.id`, `model.id`,
  `sentiment.customer_turns`, `sentiment.overall`. Never text.

## Tests

- Unit: `FakeSentimentModel` (no transformers, no torch, no network).
- Integration (`-m integration`, skipped without the `ml` group): real model on a synthetic call.
