from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from convintel.config import Settings
from convintel.llm import LLMClient, load_prompt
from convintel.llm.testing import FakeStructuredChatModel, tool_answer
from scripts.synth.judge import configured_client, judge_input, judge_sample, reviewable
from scripts.synth.review import review_batch
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import Annotation, JudgeVerdict, LabelCheck, RunSpec
from scripts.synth.storage import digest, load_samples
from scripts.synth.validation import Rejected


def run_spec(client: LLMClient) -> RunSpec:
    return RunSpec(
        seed=42,
        model_id=client.model_id,
        model_revision="test",
        judge_model_id=client.model_id,
        judge_model_revision="test",
        prompt_sha256=digest(load_prompt("synthetic", 2)),
        judge_prompt_sha256=digest(load_prompt("synthetic_judge", 1)),
        environment_sha256="b" * 64,
        pii_scan="patterns",
    )


async def test_judge_only_receives_masked_pii_and_records_binding(
    settings: Settings,
    sample_factory: Callable,
    judge_verdict: Callable,
) -> None:
    sample = sample_factory(scenarios(42)[2])
    model = FakeStructuredChatModel(
        responses=[tool_answer(JudgeVerdict, judge_verdict(sample.labels.scenario).model_dump())]
    )
    client = LLMClient(settings, chat_model=model)
    record = await judge_sample(client, sample, run_spec(client))
    prompt = " ".join(str(message.content) for message in model.calls[0])
    for value in sample.labels.pii_values:
        assert value.surface_form not in prompt
        assert value.canonical_value not in prompt
        assert value.slot.token in prompt
    assert record.input_sha256 == digest(judge_input(sample))
    assert record.verdict.passed and record.model_id == client.model_id
    assert "<conversation>" in prompt


async def test_wrong_sentiment_evidence_is_rejected(
    settings: Settings,
    sample_factory: Callable,
    judge_verdict: Callable,
) -> None:
    sample = sample_factory(scenarios(42)[0])
    verdict = judge_verdict(sample.labels.scenario).model_copy(
        update={"start_sentiment": LabelCheck(matches=True, evidence_turns=(1,))}
    )
    model = FakeStructuredChatModel(responses=[tool_answer(JudgeVerdict, verdict.model_dump())])
    client = LLMClient(settings, chat_model=model)
    with pytest.raises(Rejected, match="invalid_evidence"):
        await judge_sample(client, sample, run_spec(client))


def test_judge_booleans_are_strict_and_positive_verdict_needs_evidence() -> None:
    with pytest.raises(ValidationError):
        LabelCheck.model_validate({"matches": "true", "evidence_turns": [0]})
    with pytest.raises(ValidationError):
        LabelCheck(matches=True, evidence_turns=())


def test_only_current_judge_passed_samples_reach_human_review(
    tmp_path: Path,
    sample_factory: Callable,
) -> None:
    sample = sample_factory(scenarios(42)[0])
    assert isinstance(sample.labels, Annotation) and sample.labels.judge is not None
    absent = sample.model_copy(update={"labels": sample.labels.model_copy(update={"judge": None})})
    failed = sample.model_copy(
        update={
            "labels": sample.labels.model_copy(
                update={
                    "judge": sample.labels.judge.model_copy(
                        update={
                            "verdict": sample.labels.judge.verdict.model_copy(
                                update={"intent": LabelCheck(matches=False, evidence_turns=(0,))}
                            )
                        }
                    )
                }
            )
        }
    )
    turns = sample.conversation.turns
    changed = sample.model_copy(
        update={
            "conversation": sample.conversation.model_copy(
                update={
                    "turns": (
                        turns[0].model_copy(update={"text": "A completely different issue."}),
                        *turns[1:],
                    )
                }
            )
        }
    )
    assert reviewable(sample)
    for unqualified in (absent, failed, changed):
        assert not reviewable(unqualified)

        def forbidden(_: str) -> str:
            raise AssertionError("Unqualified candidate reached human input")

        assert (
            review_batch(
                [unqualified],
                tmp_path / "review.jsonl",
                "human",
                ask=forbidden,
                display=lambda _: None,
            )
            == 0
        )
    assert not (tmp_path / "review.jsonl").exists()


def test_historical_candidates_remain_readable_but_ineligible() -> None:
    directory = Path(__file__).resolve().parents[2] / "data/synthetic/candidates/pilot-v1.1"
    samples = load_samples(directory)
    assert len(samples) == 10 and all(not reviewable(sample) for sample in samples)
    assert digest(samples[0]) == "49e26e9b0d90fc28c988d21bdc7380606bcd9f3e7b6b0c1d0c52913b8e0996a6"


def test_ollama_clients_disable_reasoning_without_network(settings: Settings) -> None:
    generator = configured_client(settings, "synthetic", 2)
    judge = configured_client(settings, "synthetic_judge", 1)
    assert generator.model_id == judge.model_id == "ollama:qwen3:8b"
    assert generator._chat_model is not None and judge._chat_model is not None
    assert getattr(generator._chat_model, "reasoning", None) is False
    assert getattr(judge._chat_model, "reasoning", None) is False
    assert getattr(generator._chat_model, "temperature", None) == 0.35
    assert getattr(judge._chat_model, "temperature", None) == 0
