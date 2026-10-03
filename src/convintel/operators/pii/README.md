# PII redaction (`convintel.operators.pii`)

`PIIRedactor` implements `Operator[PIIOutput]`. It is the redaction boundary: every other
operator, LLM call, log and trace must only ever see `output.redacted_conversation`.

```python
from convintel.operators.pii import PIIRedactor

result = await PIIRedactor().run(conversation)
safe = result.output.redacted_conversation  # redacted=True, same turns, same indices
result.output.entities  # offsets into the ORIGINAL turn text
```

## What is detected

| Entity | Source | Notes |
|---|---|---|
| `PERSON` | spaCy NER (`en_core_web_lg`) | Score 0.85 |
| `PHONE_NUMBER` | Presidio (`phonenumbers`) | Score 0.4, unvalidated |
| `EMAIL_ADDRESS` | Presidio regex + TLD check | |
| `CREDIT_CARD` | Presidio regex + Luhn | |
| `IBAN_CODE` | Presidio regex + mod-97 | |
| `ACCOUNT_NUMBER` | `AccountNumberRecognizer` | 6-17 digits (optionally hyphen-grouped) **and** a keyword (`account`, `acct`, `a/c`, `member/customer id/number/no`) within 40 characters |

Each span is replaced by `<ENTITY_TYPE>`, e.g. `<PHONE_NUMBER>`. Overlapping detections are
merged into one span covering their union (Presidio would otherwise keep only the stronger one
and leak the rest), labelled with the strongest detection.

Threshold: 0.35. It favours recall; a bare 8-digit number without an account keyword (score 0.05)
is kept, e.g. order numbers.

## Runtime

- The spaCy model `en_core_web_lg` 3.8.0 is pinned in the `ml` dependency group (wheel URL
  in `pyproject.toml`, must match the spaCy minor version): install it with
  `uv sync --group ml`. The redactor never downloads anything; if the model is missing it raises
  `OperatorError("pii", "... run `uv sync --group ml`")`.
- The analyzer (spaCy pipeline) is a lazy, lock-protected per-process singleton
  (`get_analyzer`). First use takes a few seconds.
- Analysis runs in a worker thread (`asyncio.to_thread`).
- Span `operator.pii` with `conversation.id`, `operator.version`, `model.id`, `pii.entity_count`.
  Never put text in span attributes or logs.
- Only `language="en"` is supported; other languages raise `OperatorError`.

## Tests

- Unit (`tests/operators/pii/`): real Presidio with a blank spaCy pipeline; a deny-list of
  synthetic names stands in for NER. 24 labelled synthetic cases + hypothesis properties.
- Integration (`-m integration`): real `en_core_web_lg`, prints recall per entity type;
  skipped when the `ml` group is not installed.
