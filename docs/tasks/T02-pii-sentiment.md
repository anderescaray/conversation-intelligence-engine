# T02 — PII redaction and sentiment

**Lane:** B · **Depends on:** T00

## Goal
Redact PII before any model sees the text, and score sentiment per customer turn.

## Allowed paths
`src/convintel/operators/pii/`, `src/convintel/operators/sentiment/`, `tests/operators/pii/`, `tests/operators/sentiment/`

## Requirements
- `PIIRedactor` (Presidio analyzer + anonymizer) implementing `Operator[PIIOutput]`.
  Entities: PERSON, PHONE_NUMBER, EMAIL_ADDRESS, CREDIT_CARD, IBAN_CODE, plus a custom
  recognizer for account numbers (regex + context words). Replace with typed placeholders: `<PHONE_NUMBER>`.
- Returned `redacted_conversation` has `redacted=True` and identical turn structure.
- `SentimentOperator` implementing `Operator[SentimentOutput]` with a small HF model, batched over
  customer turns. The overall label comes from the trajectory (weight the last turns more, and justify the choice).
- Models load once (lazy singleton) and run inference in a thread pool so they don't block the event loop.

## Acceptance criteria
- [ ] PII: tests with ≥ 20 synthetic cases; recall on the custom test set reported in the PR.
- [ ] No original PII string appears in `redacted_conversation` (property-based test with hypothesis).
- [ ] Sentiment: unit tests with the model mocked + 1 integration test with the real model.
- [ ] Both operators fill `latency_ms`, `model_id`, `operator_version`.

## Out of scope
Intent, LLM calls, API.
