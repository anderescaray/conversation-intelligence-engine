from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from convintel.config import Settings
from convintel.errors import ConvIntelError
from convintel.llm import LLMClient
from convintel.llm.testing import FakeStructuredChatModel, tool_answer
from scripts.synth.generate import generate_run
from scripts.synth.review import decision
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import Draft, JudgeVerdict, ReviewEvent, Sample
from scripts.synth.storage import append_jsonl, digest, load_samples, read_jsonl


@pytest.mark.parametrize("custom_ledger", [False, True])
async def test_only_rejected_samples_can_be_explicitly_regenerated(
    tmp_path: Path,
    settings: Settings,
    draft_factory: Callable,
    custom_ledger: bool,
    judge_verdict: Callable,
) -> None:
    scenario = scenarios(42)[0]
    draft = draft_factory(scenario).model_dump()
    answer = tool_answer(JudgeVerdict, judge_verdict(scenario).model_dump())
    model = FakeStructuredChatModel(
        responses=[tool_answer(Draft, draft), answer, tool_answer(Draft, draft), answer]
    )
    client = LLMClient(settings, chat_model=model)
    output = tmp_path / "pilot"
    await generate_run(output, client, settings, seed=42, n=1, model_revision="test")
    original = load_samples(output)[0]
    with pytest.raises(ConvIntelError, match="human-rejected"):
        await generate_run(
            output,
            client,
            settings,
            seed=42,
            n=1,
            model_revision="test",
            resume=True,
            regenerate={0},
        )
    event = ReviewEvent(
        id=original.conversation.id,
        sample_sha256=digest(original),
        reviewer="test-human",
        action="reject",
        note="Semantic mismatch",
        at=datetime.now(UTC),
    )
    ledger = tmp_path / "custom-review.jsonl" if custom_ledger else output / "review.jsonl"
    append_jsonl(ledger, event)
    await generate_run(
        output,
        client,
        settings,
        seed=42,
        n=1,
        model_revision="test",
        resume=True,
        regenerate={0},
        review_jsonl=ledger,
    )
    updated = load_samples(output)[0]
    assert updated.labels.scenario == original.labels.scenario
    assert updated.labels.provenance.attempt == 2
    assert decision(updated, [event]) is None
    assert len(read_jsonl(output / "candidates.jsonl", Sample)) == 2
    assert len(model.calls) == 4
