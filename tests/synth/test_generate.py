from collections.abc import Callable
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY

from convintel.config import Settings
from convintel.contracts import Conversation
from convintel.errors import ConvIntelError
from convintel.llm import LLMClient
from convintel.llm.testing import FakeStructuredChatModel, no_sleep, tool_answer
from scripts.synth.generate import generate_run
from scripts.synth.judge import reviewable
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import (
    VERSION,
    Annotation,
    Attempt,
    Draft,
    JudgeObservation,
    LegacyJudgeRecord,
    ObservedFlag,
    ObservedIntent,
)
from scripts.synth.storage import load_samples, read_jsonl


async def test_structured_generation_and_safe_telemetry(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
    spans: InMemorySpanExporter,
) -> None:
    scenario = scenarios(42)[0]
    model = FakeStructuredChatModel(
        responses=[
            tool_answer(Draft, skeleton_draft(scenario).model_dump()),
            tool_answer(JudgeObservation, judge_observation(scenario).model_dump()),
        ]
    )
    client = LLMClient(settings, chat_model=model, sleep=no_sleep)
    directory = tmp_path / "pilot"
    result = await generate_run(directory, client, settings, seed=42, n=1, model_revision="test")
    assert (result.candidates, result.rejects, result.cost_usd) == (1, 0, 0)
    assert result.input_tokens == 200
    sample = load_samples(directory)[0]
    assert sample.labels.status == "candidate"
    assert sample.labels.scenario == scenario
    assert sample.labels.provenance.prompt_version == "synthetic/v5"
    assert isinstance(sample.labels, Annotation) and sample.labels.judge is not None
    assert sample.labels.judge.prompt_version == "synthetic_judge/v4"
    assert reviewable(sample)
    assert read_jsonl(directory / "conversations.jsonl", Conversation) == [sample.conversation]
    assert scenario.model_dump_json() in str(model.calls[0][1].content)
    assert {s.name for s in spans.get_finished_spans()} >= {
        "synthetic.generate",
        "synthetic.validate",
        "synthetic.pii",
        "synthetic.judge",
    }
    assert REGISTRY.get_sample_value(
        "convintel_operator_latency_seconds_count",
        {"operator": "synthetic.judge", "version": VERSION},
    )
    assert all("text" not in str(s.attributes) for s in spans.get_finished_spans())
    again = await generate_run(
        directory, client, settings, seed=42, n=1, model_revision="test", resume=True
    )
    assert again == result
    assert len(model.calls) == 2
    with pytest.raises(ConvIntelError, match="differs"):
        await generate_run(
            directory, client, settings, seed=43, n=1, model_revision="test", resume=True
        )


async def test_validator_rejects_preserve_labels_and_never_call_judge(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
) -> None:
    scenario = scenarios(42)[0]
    valid = skeleton_draft(scenario).model_dump()
    bad = skeleton_draft(scenario).model_dump()
    bad["turns"][0]["text"] = "Write to invented@example.com"
    model = FakeStructuredChatModel(
        responses=[
            tool_answer(Draft, bad),
            tool_answer(Draft, valid),
            tool_answer(JudgeObservation, judge_observation(scenario).model_dump()),
        ]
    )
    result = await generate_run(
        tmp_path / "pilot",
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=1,
        model_revision="test",
    )
    assert result.attempts == 2 and result.reject_rate == 0.5
    assert result.input_tokens == 300
    assert result.rejection_reasons == {"unowned_pii": 1}
    assert result.rejection_categories == {"validator": 1}
    assert result.rejection_rates == {"validator": 0.5}
    assert model.calls[0][1].content == model.calls[1][1].content
    assert "invented@example.com" not in (tmp_path / "pilot" / "attempts.jsonl").read_text()


async def test_judge_rejects_are_recorded_but_never_published(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
) -> None:
    scenario = scenarios(42)[0]
    observation = judge_observation(scenario).model_copy(
        update={
            "intent": ObservedIntent(value="exchange_rate", evidence_turns=(0,)),
            "resolved": ObservedFlag(value=not scenario.resolved, evidence_turns=(3,)),
        }
    )
    model = FakeStructuredChatModel(
        responses=[
            tool_answer(Draft, skeleton_draft(scenario).model_dump()),
            tool_answer(JudgeObservation, observation.model_dump()),
        ]
    )
    directory = tmp_path / "pilot"
    result = await generate_run(
        directory,
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=1,
        model_revision="test",
        max_attempts=1,
        max_rerolls=0,
    )
    assert result.candidates == 0
    assert result.rejection_rates == {"judge": 1.0}
    assert result.judge_failed_labels == {"intent": 1, "resolved": 1}
    assert load_samples(directory) == []
    attempt = read_jsonl(directory / "attempts.jsonl", Attempt)[0]
    assert attempt.judge is not None
    assert not isinstance(attempt.judge, LegacyJudgeRecord)
    assert attempt.judge.observation == observation
    assert attempt.judge.mismatched_labels == ("intent", "resolved")
    assert attempt.judge.model_id == "ollama:qwen3:8b"
    assert attempt.phase == "judge"


async def test_exhausted_invalid_schema_is_not_published(
    tmp_path: Path, settings: Settings
) -> None:
    model = FakeStructuredChatModel(responses=[tool_answer(Draft, {"turns": []}) for _ in range(6)])
    result = await generate_run(
        tmp_path / "pilot",
        LLMClient(settings, chat_model=model, sleep=no_sleep),
        settings,
        seed=42,
        n=1,
        model_revision="test",
        max_rerolls=0,
    )
    assert (result.candidates, result.rejects, result.input_tokens) == (0, 3, 600)
    assert result.missing_indices == [0]
    assert load_samples(tmp_path / "pilot") == []
    assert len(model.calls) == 6


async def test_scenario_length_is_enforced_in_structured_schema(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
) -> None:
    scenario = scenarios(42)[0]
    valid = skeleton_draft(scenario).model_dump()
    short = {"turns": valid["turns"][:-2]}
    model = FakeStructuredChatModel(
        responses=[
            tool_answer(Draft, short),
            tool_answer(Draft, valid),
            tool_answer(JudgeObservation, judge_observation(scenario).model_dump()),
        ]
    )
    result = await generate_run(
        tmp_path / "pilot",
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=1,
        model_revision="test",
        max_attempts=1,
        max_rerolls=0,
    )
    assert (result.candidates, result.attempts, result.input_tokens) == (1, 1, 300)
    assert len(model.calls) == 3


async def test_unreported_paid_provider_usage_is_unknown(
    tmp_path: Path, settings: Settings
) -> None:
    paid = settings.model_copy(
        update={"llm_provider": "anthropic", "llm_model": "claude-haiku-4-5"}
    )
    model = FakeStructuredChatModel(responses=[TimeoutError()])
    result = await generate_run(
        tmp_path / "pilot",
        LLMClient(paid, chat_model=model, max_retries=0),
        paid,
        seed=42,
        n=1,
        model_revision="test",
        max_attempts=1,
        max_rerolls=0,
    )
    assert result.candidates == 0 and result.cost_usd is None
    assert result.rejection_categories == {"timeout": 1}
    assert result.timed_out_ids == [scenarios(42)[0].id]
