import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from convintel.config import Settings
from convintel.errors import ConvIntelError
from convintel.llm import LLMClient
from convintel.llm.testing import FakeStructuredChatModel, tool_answer
from scripts.synth.generate import generate_run
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import Attempt, Draft, JudgeObservation
from scripts.synth.storage import load_samples, read_jsonl


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


async def test_judge_timeout_skips_sample_and_resume_does_not_restart_budget(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
) -> None:
    scenario = scenarios(42)[0]
    model = SlowFake(
        delays=[0, 1],
        responses=[
            tool_answer(Draft, skeleton_draft(scenario).model_dump()),
            tool_answer(JudgeObservation, judge_observation(scenario).model_dump()),
        ],
    )
    client = LLMClient(settings, chat_model=model)
    directory = tmp_path / "pilot"
    result = await generate_run(
        directory, client, settings, seed=42, n=1, model_revision="test", sample_timeout_s=0.3
    )
    assert result.candidates == 0 and result.attempts == 1
    assert result.rejection_categories == {"timeout": 1}
    assert result.missing_indices == [0] and result.timed_out_ids == [scenario.id]
    assert read_jsonl(directory / "attempts.jsonl", Attempt)[0].phase == "judge"
    assert load_samples(directory) == [] and len(model.calls) == 1
    resumed = await generate_run(
        directory,
        client,
        settings,
        seed=42,
        n=1,
        model_revision="test",
        sample_timeout_s=0.3,
        resume=True,
    )
    assert resumed == result and len(model.calls) == 1


async def test_validator_retries_share_one_sample_deadline(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
) -> None:
    scenario = scenarios(42)[0]
    bad = skeleton_draft(scenario).model_dump()
    bad["turns"][0]["text"] = "Email invented@example.com"
    model = SlowFake(
        delays=[0.35, 0.35],
        responses=[tool_answer(Draft, bad), tool_answer(Draft, bad)],
    )
    result = await generate_run(
        tmp_path / "pilot",
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=1,
        model_revision="test",
        sample_timeout_s=0.6,
    )
    assert result.attempts == 2
    assert result.rejection_categories == {"validator": 1, "timeout": 1}
    assert len(model.calls) == 1


async def test_sample_timeout_cannot_be_relaxed_above_120(
    tmp_path: Path, settings: Settings
) -> None:
    client = LLMClient(settings, chat_model=FakeStructuredChatModel(responses=[]))
    with pytest.raises(ConvIntelError, match="120"):
        await generate_run(
            tmp_path / "pilot",
            client,
            settings,
            seed=42,
            n=1,
            model_revision="test",
            sample_timeout_s=121,
        )
    assert not (tmp_path / "pilot").exists()


async def test_timeout_continues_with_next_sample(
    tmp_path: Path,
    settings: Settings,
    skeleton_draft: Callable,
    judge_observation: Callable,
) -> None:
    scenario = scenarios(42)[1]
    model = SlowFake(
        delays=[1, 0, 0],
        responses=[
            tool_answer(Draft, skeleton_draft(scenario).model_dump()),
            tool_answer(JudgeObservation, judge_observation(scenario).model_dump()),
        ],
    )
    result = await generate_run(
        tmp_path / "pilot",
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=2,
        model_revision="test",
        sample_timeout_s=0.3,
    )
    assert result.candidates == 1 and result.attempts == 2
    assert result.missing_indices == [0]
    assert load_samples(tmp_path / "pilot")[0].conversation.id == scenario.id
    assert len(model.calls) == 2
