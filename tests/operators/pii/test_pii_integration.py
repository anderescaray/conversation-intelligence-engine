"""Real spaCy NER (`en_core_web_lg`, `uv sync --group ml`): recall on the synthetic set."""

from collections import Counter

import pytest
from pii_synthetic_cases import CASES, make_conversation, recalled

from convintel.operators.pii import PIIRedactor

pytestmark = pytest.mark.integration

MIN_RECALL = 0.9


async def test_recall_with_real_ner(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("en_core_web_lg")
    redactor = PIIRedactor()
    found: Counter[str] = Counter()
    total: Counter[str] = Counter()
    kept_ok = 0
    must_keep = 0
    for case in CASES:
        result = await redactor.run(make_conversation(case.text))
        redacted = result.output.redacted_conversation.turns[0].text
        for (entity_type, _), hit in zip(
            case.expected, recalled(case, result.output.entities, redacted), strict=True
        ):
            total[entity_type] += 1
            found[entity_type] += hit
        must_keep += len(case.must_keep)
        kept_ok += sum(keep in redacted for keep in case.must_keep)

    recall = sum(found.values()) / sum(total.values())
    with capsys.disabled():
        print(f"\nPII recall on {len(CASES)} synthetic cases: {recall:.3f}")
        for entity_type in sorted(total):
            print(f"  {entity_type:<15} {found[entity_type]}/{total[entity_type]}")
        print(f"  non-PII kept   {kept_ok}/{must_keep}")
    assert recall >= MIN_RECALL
