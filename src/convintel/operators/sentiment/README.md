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
  the first conversation with customer turns; inference runs in a worker thread.
- Needs `torch` at load time. torch is only in the `ml` group today (ADR-0005), so the
  production image must add a CPU or CUDA build (T13).
- Span `operator.sentiment` with `conversation.id`, `operator.version`, `model.id`,
  `sentiment.customer_turns`, `sentiment.overall`.

## Tests

- Unit: `FakeSentimentModel` (no transformers, no torch, no network).
- Integration (`-m integration`, skipped without torch): real model on a synthetic call.
