from __future__ import annotations

import socket
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from convintel.config import Settings, get_settings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if TYPE_CHECKING:
    from scripts.synth.schema import Draft, JudgeVerdict, Sample, Scenario


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Synthetic tests must never access the network")

    for name in ("LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_API_KEY", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    options: dict[str, Any] = {"_env_file": None, "llm_provider": "ollama", "llm_model": "qwen3:8b"}
    return Settings(**options)


@pytest.fixture
def draft_factory() -> Callable[[Scenario], Draft]:
    from scripts.synth.schema import Draft, DraftTurn

    def factory(scenario: Scenario) -> Draft:
        texts = [
            "I need help with my banking issue.",
            "I can check the account information.",
            "Please explain what has happened to my request.",
            "The next step is clear and I can explain the outcome.",
            "I understand the outcome and have no more questions.",
            "I have recorded our discussion for reference.",
            "I appreciate that explanation of the remaining work.",
            "You can contact the support team if anything changes.",
            "That answers my last question about the process.",
            "Thank you for discussing the options with me.",
        ][: scenario.customer_turns * 2]
        if scenario.escalation_requested:
            texts[0] += " I am frustrated and want to speak to a supervisor."
        for slot in scenario.slots:
            context = (
                f"My {slot.entity_type.lower().replace('_', ' ')} is"
                if slot.exposure == "full"
                else "I confirm the last four digits"
            )
            texts[slot.turn_index] += f" {context} {slot.token}."
        return Draft(
            turns=[
                DraftTurn(
                    speaker="customer" if i % 2 == 0 else "agent",
                    text=text,
                )
                for i, text in enumerate(texts)
            ]
        )

    return factory


@pytest.fixture
def judge_verdict() -> Callable[[Scenario], JudgeVerdict]:
    from scripts.synth.schema import JudgeVerdict, LabelCheck

    def factory(scenario: Scenario) -> JudgeVerdict:
        return JudgeVerdict(
            intent=LabelCheck(matches=True, evidence_turns=(0,)),
            resolved=LabelCheck(matches=True, evidence_turns=(3,)),
            escalation_requested=LabelCheck(matches=True, evidence_turns=(0,)),
            start_sentiment=LabelCheck(matches=True, evidence_turns=(0,)),
            end_sentiment=LabelCheck(
                matches=True, evidence_turns=(2 * scenario.customer_turns - 2,)
            ),
        )

    return factory


@pytest.fixture
def with_judge(judge_verdict: Callable[[Scenario], JudgeVerdict]) -> Callable[[Sample], Sample]:
    from scripts.synth.judge import judge_input
    from scripts.synth.schema import Annotation, JudgeRecord
    from scripts.synth.storage import digest

    def attach(sample: Sample) -> Sample:
        assert isinstance(sample.labels, Annotation)
        verdict = JudgeRecord(
            model_id="ollama:qwen3:8b",
            model_revision="test",
            prompt_version="synthetic_judge/v1",
            prompt_sha256="c" * 64,
            input_sha256=digest(judge_input(sample)),
            temperature=0,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0,
            latency_ms=1,
            verdict=judge_verdict(sample.labels.scenario),
        )
        return sample.model_copy(
            update={"labels": sample.labels.model_copy(update={"judge": verdict})}
        )

    return attach


@pytest.fixture
def sample_factory(
    draft_factory: Callable[[Scenario], Draft],
    with_judge: Callable[[Sample], Sample],
) -> Callable[[Scenario], Sample]:
    from scripts.synth.pii import insert_pii
    from scripts.synth.schema import Annotation, GenerationProvenance, Sample

    def factory(scenario: Scenario) -> Sample:
        conversation, values, entities = insert_pii(draft_factory(scenario), scenario)
        sample = Sample(
            conversation=conversation,
            labels=Annotation(
                id=scenario.id,
                scenario=scenario,
                pii_values=values,
                pii_entities=entities,
                provenance=GenerationProvenance(
                    model_id="ollama:qwen3:8b",
                    model_revision="test",
                    prompt_version="synthetic/v2",
                    prompt_sha256="a" * 64,
                    environment_sha256="b" * 64,
                    temperature=0.35,
                    attempt=1,
                    input_tokens=100,
                    output_tokens=20,
                    cost_usd=0,
                    latency_ms=1,
                    pii_scan="patterns",
                ),
            ),
        )
        return with_judge(sample)

    return factory


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    yield exporter
    provider.shutdown()
