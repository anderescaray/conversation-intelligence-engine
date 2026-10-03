import threading
from collections import Counter
from typing import Any

import pytest
from pii_synthetic_cases import CASES, PIICase, make_conversation, recalled
from presidio_analyzer import AnalyzerEngine, RecognizerResult

from convintel.contracts import Operator, PIIOutput, Speaker
from convintel.errors import OperatorError
from convintel.operators.pii import PIIRedactor, engine, merge_overlaps
from convintel.operators.pii import redactor as redactor_module


def test_synthetic_set_has_at_least_20_cases() -> None:
    assert len(CASES) >= 20
    assert sum(len(c.expected) for c in CASES) >= 20


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
async def test_synthetic_case_is_redacted(redactor: PIIRedactor, case: PIICase) -> None:
    result = await redactor.run(make_conversation(case.text))

    redacted = result.output.redacted_conversation.turns[0].text
    assert all(recalled(case, result.output.entities, redacted))
    for entity_type, value in case.expected:
        assert value not in redacted
        assert f"<{entity_type}>" in redacted
    for keep in case.must_keep:
        assert keep in redacted
    if not case.expected:
        assert result.output.entities == []


async def test_recall_on_synthetic_set(redactor: PIIRedactor) -> None:
    """Recall per entity type with the NER stubbed; the integration test measures real NER."""
    found: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for case in CASES:
        result = await redactor.run(make_conversation(case.text))
        redacted = result.output.redacted_conversation.turns[0].text
        for (entity_type, _), hit in zip(
            case.expected, recalled(case, result.output.entities, redacted), strict=True
        ):
            total[entity_type] += 1
            found[entity_type] += hit

    assert sum(found.values()) / sum(total.values()) == 1.0


async def test_keeps_turn_structure_and_marks_redacted(redactor: PIIRedactor) -> None:
    conversation = make_conversation(
        "Hi, this is Tobias Wrenfield.",
        "Thanks, how can I help?",
        "Call me on 415-555-0123.",
    )

    result = await redactor.run(conversation)

    redacted = result.output.redacted_conversation
    assert redacted.redacted is True
    assert redacted.id == conversation.id
    assert len(redacted.turns) == len(conversation.turns)
    for before, after in zip(conversation.turns, redacted.turns, strict=True):
        assert (after.index, after.speaker) == (before.index, before.speaker)
    assert redacted.turns[1].text == "Thanks, how can I help?"
    assert redacted.turns[2].text == "Call me on <PHONE_NUMBER>."


async def test_does_not_mutate_input(redactor: PIIRedactor) -> None:
    conversation = make_conversation("Mail marisol.q@example.com")

    await redactor.run(conversation)

    assert conversation.redacted is False
    assert conversation.turns[0].text == "Mail marisol.q@example.com"


async def test_entity_offsets_point_into_original_turn(redactor: PIIRedactor) -> None:
    conversation = make_conversation("ok", "Hi Lucas Fernhill, your acct 0042817365 is fine.")

    result = await redactor.run(conversation)

    by_type = {e.entity_type: e for e in result.output.entities}
    assert set(by_type) == {"PERSON", "ACCOUNT_NUMBER"}
    text = conversation.turns[1].text
    assert all(e.turn_index == 1 for e in result.output.entities)
    assert text[by_type["PERSON"].start : by_type["PERSON"].end] == "Lucas Fernhill"
    account = by_type["ACCOUNT_NUMBER"]
    assert text[account.start : account.end] == "0042817365"


async def test_result_envelope(redactor: PIIRedactor) -> None:
    result = await redactor.run(make_conversation("nothing to see"))

    assert isinstance(redactor, Operator)
    assert isinstance(result.output, PIIOutput)
    assert result.operator == "pii"
    assert result.operator_version == redactor.version
    assert result.model_id == "presidio/blank_en"
    assert result.latency_ms >= 0
    assert result.cost_usd is None


async def test_unsupported_language_raises_operator_error(redactor: PIIRedactor) -> None:
    conversation = make_conversation("hola").model_copy(update={"language": "xx"})

    with pytest.raises(OperatorError) as excinfo:
        await redactor.run(conversation)

    assert excinfo.value.operator == "pii"


async def test_analysis_runs_off_the_event_loop_thread(
    analyzer: AnalyzerEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[int] = []
    analyze = analyzer.analyze

    def spy(*args: Any, **kwargs: Any) -> list[RecognizerResult]:
        threads.append(threading.get_ident())
        return analyze(*args, **kwargs)

    monkeypatch.setattr(analyzer, "analyze", spy)

    await PIIRedactor(analyzer=analyzer).run(make_conversation("hello"))

    assert threads
    assert threading.get_ident() not in threads


async def test_uses_the_shared_analyzer_when_none_is_injected(
    analyzer: AnalyzerEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[str] = []

    def fake_get_analyzer(spacy_model: str) -> AnalyzerEngine:
        requested.append(spacy_model)
        return analyzer

    monkeypatch.setattr(redactor_module, "get_analyzer", fake_get_analyzer)
    redactor = PIIRedactor(spacy_model="en_core_web_sm")

    await redactor.run(make_conversation("Call 415-555-0123"))
    await redactor.run(make_conversation("Call 415-555-0123"))

    assert requested == ["en_core_web_sm", "en_core_web_sm"]


def test_get_analyzer_builds_once_under_concurrency(
    analyzer: AnalyzerEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds: list[str] = []

    def fake_build(spacy_model: str) -> AnalyzerEngine:
        builds.append(spacy_model)
        return analyzer

    monkeypatch.setattr(engine, "build_analyzer", fake_build)
    engine._cached_analyzer.cache_clear()  # pyright: ignore[reportPrivateUsage]
    try:
        got: list[AnalyzerEngine] = []
        workers = [
            threading.Thread(target=lambda: got.append(engine.get_analyzer("m"))) for _ in range(8)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    finally:
        engine._cached_analyzer.cache_clear()  # pyright: ignore[reportPrivateUsage]

    assert builds == ["m"]
    assert len(got) == 8
    assert all(a is analyzer for a in got)


async def test_agent_turns_are_redacted_too(redactor: PIIRedactor) -> None:
    """Agents read PII back to customers, so redaction ignores the speaker."""
    conversation = make_conversation("hi", "I have 415-555-0123 on file, correct?")
    assert conversation.turns[1].speaker is Speaker.AGENT

    result = await redactor.run(conversation)

    assert result.output.redacted_conversation.turns[1].text == (
        "I have <PHONE_NUMBER> on file, correct?"
    )


class TestMergeOverlaps:
    def test_disjoint_spans_are_kept_in_order(self) -> None:
        spans = [
            RecognizerResult("EMAIL_ADDRESS", 20, 30, 1.0),
            RecognizerResult("PERSON", 0, 5, 0.85),
        ]

        merged = merge_overlaps(spans)

        assert [(m.entity_type, m.start, m.end) for m in merged] == [
            ("PERSON", 0, 5),
            ("EMAIL_ADDRESS", 20, 30),
        ]

    def test_overlap_covers_union_and_keeps_strongest_label(self) -> None:
        spans = [
            RecognizerResult("PHONE_NUMBER", 10, 22, 0.4),
            RecognizerResult("ACCOUNT_NUMBER", 6, 22, 0.85),
            RecognizerResult("CREDIT_CARD", 18, 30, 0.3),
        ]

        merged = merge_overlaps(spans)

        assert len(merged) == 1
        assert (merged[0].entity_type, merged[0].start, merged[0].end) == ("ACCOUNT_NUMBER", 6, 30)
        assert merged[0].score == 0.85

    def test_adjacent_spans_stay_separate(self) -> None:
        spans = [RecognizerResult("PERSON", 0, 5, 0.9), RecognizerResult("PERSON", 5, 9, 0.9)]

        assert len(merge_overlaps(spans)) == 2

    def test_does_not_mutate_input(self) -> None:
        first = RecognizerResult("PHONE_NUMBER", 0, 10, 0.4)

        merge_overlaps([first, RecognizerResult("CREDIT_CARD", 5, 15, 1.0)])

        assert (first.entity_type, first.end, first.score) == ("PHONE_NUMBER", 10, 0.4)
