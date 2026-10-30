import asyncio
import threading
import time
import traceback
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
import spacy.cli
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
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


async def test_missing_spacy_model_raises_instead_of_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("PIIRedactor must never download a model")

    monkeypatch.setattr(spacy.cli, "download", no_download)
    redactor = PIIRedactor(spacy_model="en_core_web_not_installed")

    with pytest.raises(OperatorError, match=r"uv sync --group ml") as excinfo:
        await redactor.run(make_conversation("hello"))

    assert excinfo.value.operator == "pii"
    assert "en_core_web_not_installed" in excinfo.value.message


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


class _RecordingTracer:
    """Records exceptions the way the OTel SDK does: the full traceback, causes included."""

    def __init__(self) -> None:
        self.recorded: list[str] = []

    @contextmanager
    def start_as_current_span(self, name: str) -> Iterator[MagicMock]:
        try:
            yield MagicMock()
        except BaseException as exc:
            self.recorded.append("".join(traceback.format_exception(exc)))
            raise


async def test_errors_never_carry_turn_text(
    analyzer: AnalyzerEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "marisol.q@example.com"

    def failing_analyze(text: str, **kwargs: Any) -> list[RecognizerResult]:
        raise ValueError(f"cannot parse {text!r}")

    tracer = _RecordingTracer()  # stands in for the span the pipeline opens around run()
    monkeypatch.setattr(analyzer, "analyze", failing_analyze)

    with pytest.raises(OperatorError) as excinfo, tracer.start_as_current_span("operator.pii"):
        await PIIRedactor(analyzer=analyzer).run(make_conversation(f"mail {secret}"))

    error = excinfo.value
    assert error.message == "redaction failed (ValueError)"
    assert error.__cause__ is None
    assert error.__suppress_context__
    assert secret not in "".join(traceback.format_exception(error))
    assert len(tracer.recorded) == 1
    assert secret not in tracer.recorded[0]


async def test_run_annotates_the_callers_span(analyzer: AnalyzerEngine) -> None:
    # The pipeline opens operator.pii with traced(); a span opened inside run() would
    # double-count latency and errors, and these attributes would land on it instead.
    secret = "marisol.q@example.com"
    parent = MagicMock(spec=trace.Span)
    with trace.use_span(parent, end_on_exit=False):
        await PIIRedactor(analyzer=analyzer).run(make_conversation(f"mail {secret}"))

    attributes = {c.args[0]: c.args[1] for c in parent.set_attribute.call_args_list}
    assert attributes["pii.entity_count"] >= 1
    assert str(attributes["model.id"]).startswith("presidio/")
    assert secret not in repr(attributes)


async def test_concurrent_runs_are_serialized(
    analyzer: AnalyzerEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared analyzer is not thread-safe; to_thread alone would run these in parallel."""
    active = 0
    peak = 0
    calls = 0
    counter_lock = threading.Lock()
    analyze = analyzer.analyze

    def slow_analyze(*args: Any, **kwargs: Any) -> list[RecognizerResult]:
        nonlocal active, peak, calls
        with counter_lock:
            active += 1
            calls += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with counter_lock:
            active -= 1
        return analyze(*args, **kwargs)

    monkeypatch.setattr(analyzer, "analyze", slow_analyze)
    redactors = [PIIRedactor(analyzer=analyzer) for _ in range(6)]

    await asyncio.gather(*(r.run(make_conversation("hello")) for r in redactors))

    assert calls == 6
    assert peak == 1


async def test_warmup_loads_the_analyzer_once_and_run_reuses_it(
    analyzer: AnalyzerEngine, monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    requested: list[str] = []
    analyzed: list[str] = []
    analyze = analyzer.analyze

    def spy(text: str, **kwargs: Any) -> list[RecognizerResult]:
        analyzed.append(text)
        return analyze(text, **kwargs)

    def load_once(spacy_model: str) -> AnalyzerEngine:
        requested.append(spacy_model)
        return analyzer

    monkeypatch.setattr(analyzer, "analyze", spy)
    monkeypatch.setattr(redactor_module, "get_analyzer", load_once)
    redactor = PIIRedactor(spacy_model="en_core_web_sm")

    await redactor.warmup()
    monkeypatch.setattr(redactor_module, "get_analyzer", _never_load)
    result = await redactor.run(make_conversation("Call 415-555-0123"))

    assert requested == ["en_core_web_sm"]
    assert analyzed == [redactor_module._WARMUP_TEXT, "Call 415-555-0123"]  # pyright: ignore[reportPrivateUsage]
    assert result.output.redacted_conversation.turns[0].text == "Call <PHONE_NUMBER>"
    (span,) = [s for s in spans.get_finished_spans() if s.name == "operator.pii.warmup"]
    attributes = span.attributes or {}
    assert attributes["model.id"] == "presidio/en_core_web_sm"
    assert attributes["operator.version"] == redactor.version
    assert span.status.status_code is not StatusCode.ERROR


async def test_warmup_text_has_no_pii(redactor: PIIRedactor) -> None:
    # Warmup spans and errors may carry it; it must stay free of anything to redact.
    text = redactor_module._WARMUP_TEXT  # pyright: ignore[reportPrivateUsage]
    result = await redactor.run(make_conversation(text))

    assert result.output.entities == []


async def test_warmup_without_the_spacy_model_fails_and_marks_the_span(
    spans: InMemorySpanExporter,
) -> None:
    with pytest.raises(OperatorError, match=r"uv sync --group ml") as excinfo:
        await PIIRedactor(spacy_model="en_core_web_not_installed").warmup()

    assert excinfo.value.operator == "pii"
    (span,) = [s for s in spans.get_finished_spans() if s.name == "operator.pii.warmup"]
    assert span.status.status_code is StatusCode.ERROR
    assert (span.attributes or {})["error.type"] == "OperatorError"
    assert not span.events


async def test_warmup_failures_keep_only_the_error_type(
    analyzer: AnalyzerEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_analyze(text: str, **kwargs: Any) -> list[RecognizerResult]:
        raise ValueError(f"cannot parse {text!r}")

    monkeypatch.setattr(analyzer, "analyze", failing_analyze)

    with pytest.raises(OperatorError) as excinfo:
        await PIIRedactor(analyzer=analyzer).warmup()

    assert excinfo.value.message == "warmup failed (ValueError)"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__


def _never_load(spacy_model: str) -> AnalyzerEngine:
    raise AssertionError("run() after warmup() must reuse the warmed analyzer")
