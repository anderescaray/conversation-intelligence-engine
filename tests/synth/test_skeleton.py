import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from convintel.config import Settings
from convintel.errors import ConvIntelError
from convintel.llm import LLMClient
from convintel.llm.testing import FakeStructuredChatModel, tool_answer
from scripts.synth.generate import generate_run
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import (
    VERSION,
    Annotation,
    Attempt,
    Draft,
    IndexedDraftTurn,
    JudgeObservation,
    RunSpec,
    SkeletonDraft,
    skeleton,
)
from scripts.synth.storage import load_samples, read_jsonl
from scripts.synth.validation import skeleton_errors

LEGACY_RUN = Path(__file__).resolve().parents[2] / "data/synthetic/candidates/pilot-v2/run.json"


class SlowFake(FakeStructuredChatModel):
    delays: list[float]

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        await asyncio.sleep(self.delays.pop(0))
        return self._generate(messages, stop, None, **kwargs)


def moved_token(draft: SkeletonDraft, token: str, destination: int) -> SkeletonDraft:
    turns = [
        turn.model_copy(update={"text": turn.text.replace(f" {token}", "")}) for turn in draft.turns
    ]
    turns[destination] = turns[destination].model_copy(
        update={"text": f"{turns[destination].text} {token}"}
    )
    return SkeletonDraft(turns=turns)


def with_extra(draft: SkeletonDraft, index: int, token: str) -> SkeletonDraft:
    turns = list(draft.turns)
    turns[index] = turns[index].model_copy(update={"text": f"{turns[index].text} {token}"})
    return SkeletonDraft(turns=turns)


def test_skeleton_fixes_speakers_placeholders_and_cues() -> None:
    scenario = scenarios(42)[31]
    plan = skeleton(scenario)
    assert len(plan) == 2 * scenario.customer_turns
    assert [turn.speaker for turn in plan] == ["customer", "agent"] * scenario.customer_turns
    assert [turn.placeholders for turn in plan[:4]] == [
        ("[[PHONE_NUMBER_1]]",),
        ("[[PHONE_NUMBER_LAST_FOUR_1]]",),
        ("[[CREDIT_CARD_1]]",),
        (),
    ]
    assert "last four digits" in plan[1].cues[0]
    assert plan[0].cues[0].endswith(scenario.start_sentiment.value)
    assert any(cue.endswith(scenario.end_sentiment.value) for cue in plan[-2].cues)
    assert all(not turn.placeholders for turn in plan[4:])


def test_skeleton_errors_name_every_concrete_mismatch(skeleton_draft: Callable) -> None:
    scenario = scenarios(42)[1]
    plan = skeleton(scenario)
    valid = skeleton_draft(scenario)
    assert skeleton_errors(valid, plan) == []
    moved = moved_token(valid, "[[EMAIL_ADDRESS_1]]", 1)
    assert skeleton_errors(moved, plan) == [
        "turn 0 (customer) must contain [[EMAIL_ADDRESS_1]] exactly once; found 0",
        "turn 1 (agent) must not contain [[EMAIL_ADDRESS_1]]",
    ]
    turns = list(valid.turns)
    turns[2] = turns[2].model_copy(update={"text": turns[2].text + " [[IBAN_CODE_1]]"})
    turns[4] = IndexedDraftTurn(turn_index=7, speaker="agent", text="Fine.")
    assert skeleton_errors(SkeletonDraft(turns=turns[:5] + turns[:1]), plan) == [
        "turn 2 (customer) must contain [[IBAN_CODE_1]] exactly once; found 2",
        "entry 4 must have turn_index 4",
        "turn 4 must be spoken by the customer",
        "entry 5 must have turn_index 5",
        "turn 5 must be spoken by the agent",
        "turn 5 (agent) must not contain [[EMAIL_ADDRESS_1]]",
    ]
    longer = skeleton(scenarios(42)[2])
    assert skeleton_errors(valid, longer)[0] == "expected exactly 8 turns, got 6"


async def test_one_repair_with_concrete_errors_then_judge(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
    spans: InMemorySpanExporter,
) -> None:
    scenario = scenarios(42)[0]
    valid = skeleton_draft(scenario)
    bad = with_extra(valid, 1, "[[PERSON_1]]")
    model = FakeStructuredChatModel(
        responses=[
            tool_answer(Draft, bad.model_dump()),
            tool_answer(Draft, valid.model_dump()),
            tool_answer(JudgeObservation, judge_observation(scenario).model_dump()),
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
    assert result.candidates == 1 and result.missing_indices == []
    assert (result.repairs, result.repairs_fixed) == (1, 1)
    repair_prompt = str(model.calls[1][1].content)
    assert "<repair>" in repair_prompt and "<repair>" not in str(model.calls[0][1].content)
    assert "turn 1 (agent) must not contain [[PERSON_1]]" in repair_prompt
    attempt = read_jsonl(directory / "attempts.jsonl", Attempt)[0]
    assert attempt.accepted and attempt.repaired
    assert attempt.generation_usage is not None and attempt.judge_usage is not None
    assert attempt.generation_usage.input_tokens == 200
    assert attempt.judge_usage.input_tokens == 100
    sample = load_samples(directory)[0]
    assert isinstance(sample.labels, Annotation)
    assert sample.labels.provenance.repaired and sample.labels.provenance.input_tokens == 200
    assert sample.labels.provenance.prompt_version == "synthetic/v3"
    assert "synthetic.repair" in {s.name for s in spans.get_finished_spans()}


async def test_failed_repair_rejects_without_judging(
    tmp_path: Path, settings: Settings, skeleton_draft: Callable
) -> None:
    scenario = scenarios(42)[0]
    bad = with_extra(skeleton_draft(scenario), 1, "[[PERSON_1]]").model_dump()
    model = FakeStructuredChatModel(responses=[tool_answer(Draft, bad), tool_answer(Draft, bad)])
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
    assert (result.repairs, result.repairs_fixed) == (1, 0)
    attempt = read_jsonl(directory / "attempts.jsonl", Attempt)[0]
    assert attempt.reason == "skeleton_mismatch" and attempt.category == "validator"
    assert attempt.repaired and attempt.phase == "generation" and attempt.judge_usage is None
    assert len(model.calls) == 2


async def test_repair_shares_the_sample_deadline(
    tmp_path: Path, settings: Settings, skeleton_draft: Callable
) -> None:
    scenario = scenarios(42)[0]
    valid = skeleton_draft(scenario)
    bad = with_extra(valid, 1, "[[PERSON_1]]")
    model = SlowFake(
        delays=[0.2, 0.2],
        responses=[tool_answer(Draft, bad.model_dump()), tool_answer(Draft, valid.model_dump())],
    )
    directory = tmp_path / "pilot"
    result = await generate_run(
        directory,
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=1,
        model_revision="test",
        sample_timeout_s=0.3,
    )
    attempts = read_jsonl(directory / "attempts.jsonl", Attempt)
    assert len(attempts) == 1 and len(model.calls) == 1
    assert attempts[0].category == "timeout" and attempts[0].repaired
    assert result.timed_out_ids == [scenario.id]


async def test_judge_is_independent_of_generator_and_costs_are_split(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
) -> None:
    scenario = scenarios(42)[0]
    generator_model = FakeStructuredChatModel(
        responses=[tool_answer(Draft, skeleton_draft(scenario).model_dump())]
    )
    judge_model = FakeStructuredChatModel(
        responses=[
            tool_answer(
                JudgeObservation,
                judge_observation(scenario).model_dump(),
                input_tokens=1000,
                output_tokens=100,
            )
        ]
    )
    paid = settings.model_copy(
        update={"llm_provider": "anthropic", "llm_model": "claude-haiku-4-5"}
    )
    directory = tmp_path / "pilot"
    result = await generate_run(
        directory,
        LLMClient(settings, chat_model=generator_model),
        settings,
        judge_client=LLMClient(paid, chat_model=judge_model),
        seed=42,
        n=1,
        model_revision="qwen-digest",
        judge_model_revision="haiku-snapshot",
    )
    run = RunSpec.model_validate_json((directory / "run.json").read_text())
    assert run.model_id == "ollama:qwen3:8b" and run.generator_version == VERSION == "3.1.0"
    assert (run.judge_model_id, run.judge_model_revision) == (
        "anthropic:claude-haiku-4-5",
        "haiku-snapshot",
    )
    assert (result.model_id, result.judge_model_id) == (run.model_id, run.judge_model_id)
    assert len(generator_model.calls) == len(judge_model.calls) == 1
    generation, judge = result.phase_totals["generation"], result.phase_totals["judge"]
    assert (generation.input_tokens, generation.cost_usd) == (100, 0)
    assert judge.input_tokens == 1000
    assert judge.cost_usd == pytest.approx(1000 / 1e6 * 1.0 + 100 / 1e6 * 5.0)
    assert result.phase_per_admitted["judge"] == judge
    assert result.phase_mean_per_scenario["generation"] == generation
    assert result.cost_usd == pytest.approx(judge.cost_usd)
    sample = load_samples(directory)[0]
    assert isinstance(sample.labels, Annotation) and sample.labels.judge is not None
    assert sample.labels.judge.model_id == "anthropic:claude-haiku-4-5"
    assert sample.labels.judge.reasoning is None


async def test_resume_refuses_runs_from_generator_two(tmp_path: Path, settings: Settings) -> None:
    directory = tmp_path / "pilot"
    await asyncio.to_thread(directory.mkdir)
    await asyncio.to_thread(shutil.copy, LEGACY_RUN, directory / "run.json")
    client = LLMClient(settings, chat_model=FakeStructuredChatModel(responses=[]))
    with pytest.raises(ConvIntelError, match="differs"):
        await generate_run(
            directory, client, settings, seed=42, n=1, model_revision="test", resume=True
        )
