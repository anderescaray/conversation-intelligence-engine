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

- The model is a lazy, lock-protected per-process singleton (`get_sentiment_model`), loaded on
  the first conversation with customer turns. Inference runs in a worker thread under a
  process-wide lock (transformers pipelines are not thread-safe), so one batch runs at a time
  per process: **scale with worker processes**, not threads.
- Needs `transformers` and `torch`, both in the `ml` group (ADR-0005), imported only when the
  model loads. The production image must install them (T13). The model weights are fetched
  from the Hugging Face hub on first load and cached.
- Errors: `OperatorError("sentiment", "sentiment inference failed (<ExceptionType>)")`, with no
  chained cause, so text quoted in third-party messages never reaches spans or logs.
- Span `operator.sentiment` with `conversation.id`, `operator.version`, `model.id`,
  `sentiment.customer_turns`, `sentiment.overall`.

## Tests

- Unit: `FakeSentimentModel` (no transformers, no torch, no network).
- Integration (`-m integration`, skipped without the `ml` group): real model on a synthetic call.
