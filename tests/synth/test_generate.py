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
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import VERSION, Draft
from scripts.synth.storage import load_samples, read_jsonl


async def test_structured_generation_and_safe_telemetry(
    tmp_path: Path,
    settings: Settings,
    draft_factory: Callable,
    spans: InMemorySpanExporter,
) -> None:
    scenario = scenarios(42)[0]
    model = FakeStructuredChatModel(
        responses=[tool_answer(Draft, draft_factory(scenario).model_dump())]
    )
    client = LLMClient(settings, chat_model=model, sleep=no_sleep)
    directory = tmp_path / "pilot"
    result = await generate_run(directory, client, settings, seed=42, n=1, model_revision="test")
    assert result.candidates == 1
    assert result.rejects == 0
    assert result.cost_usd == 0
    sample = load_samples(directory)[0]
    assert sample.labels.status == "candidate"
    assert sample.labels.scenario == scenario
    assert sample.labels.provenance.prompt_version == "synthetic/v1"
    assert read_jsonl(directory / "conversations.jsonl", Conversation) == [sample.conversation]
    assert scenario.model_dump_json() in str(model.calls[0][1].content)
    assert {s.name for s in spans.get_finished_spans()} >= {
        "synthetic.generate",
        "synthetic.validate",
        "synthetic.pii",
    }
    assert REGISTRY.get_sample_value(
        "convintel_operator_latency_seconds_count",
        {"operator": "synthetic.generate", "version": VERSION},
    )
    assert all("text" not in str(s.attributes) for s in spans.get_finished_spans())
    again = await generate_run(
        directory,
        client,
        settings,
        seed=42,
        n=1,
        model_revision="test",
        resume=True,
    )
    assert again == result
    assert len(model.calls) == 1
    with pytest.raises(ConvIntelError, match="differs"):
        await generate_run(
            directory, client, settings, seed=43, n=1, model_revision="test", resume=True
        )


async def test_rejects_preserve_labels_and_count_tokens(
    tmp_path: Path,
    settings: Settings,
    draft_factory: Callable,
) -> None:
    scenario = scenarios(42)[0]
    valid = draft_factory(scenario).model_dump()
    bad = draft_factory(scenario).model_dump()
    bad["turns"][0]["text"] = "Write to invented@example.com"
    model = FakeStructuredChatModel(responses=[tool_answer(Draft, bad), tool_answer(Draft, valid)])
    result = await generate_run(
        tmp_path / "pilot",
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=1,
        model_revision="test",
    )
    assert result.attempts == 2
    assert result.reject_rate == 0.5
    assert result.input_tokens == 200
    assert result.rejection_reasons == {"unowned_pii": 1}
    assert model.calls[0][1].content == model.calls[1][1].content
    assert "invented@example.com" not in (tmp_path / "pilot" / "attempts.jsonl").read_text()


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
    )
    assert result.candidates == 0
    assert result.rejects == 3
    assert result.input_tokens == 600
    assert result.missing_indices == [0]
    assert load_samples(tmp_path / "pilot") == []
    assert len(model.calls) == 6


async def test_scenario_length_is_enforced_in_structured_schema(
    tmp_path: Path,
    settings: Settings,
    draft_factory: Callable,
) -> None:
    valid = draft_factory(scenarios(42)[0]).model_dump()
    short = {"turns": valid["turns"][:6]}
    model = FakeStructuredChatModel(
        responses=[tool_answer(Draft, short), tool_answer(Draft, valid)]
    )
    result = await generate_run(
        tmp_path / "pilot",
        LLMClient(settings, chat_model=model),
        settings,
        seed=42,
        n=1,
        model_revision="test",
        max_attempts=1,
    )
    assert result.candidates == 1
    assert result.attempts == 1
    assert result.input_tokens == 200
    assert len(model.calls) == 2


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
    )
    assert result.candidates == 0
    assert result.cost_usd is None
    assert result.rejection_reasons == {"llm_error": 1}
