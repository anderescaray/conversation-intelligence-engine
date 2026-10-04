import socket
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from convintel.config import Settings, get_settings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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
def draft_factory() -> Callable:
    from scripts.synth.schema import Draft, DraftTurn, Scenario

    def factory(scenario: Scenario) -> Draft:
        texts = [
            "I need help with my banking issue. " + " ".join(s.token for s in scenario.slots),
            "I will check the issue and explain the next step.",
            "I am still concerned about what happened.",
            "The issue is being investigated.",
            "Please ask a supervisor to help."
            if scenario.escalation_requested
            else "Thank you for checking.",
            "I will keep you updated.",
        ]
        texts.extend(
            ["Can you explain the next step?", "I will explain it now."]
            * (scenario.customer_turns - 3)
        )
        return Draft(
            turns=[
                DraftTurn(speaker="customer" if i % 2 == 0 else "agent", text=text)
                for i, text in enumerate(texts)
            ]
        )

    return factory


@pytest.fixture
def sample_factory(draft_factory: Callable) -> Callable:
    from scripts.synth.pii import insert_pii
    from scripts.synth.schema import Annotation, Provenance, Sample, Scenario

    def factory(scenario: Scenario) -> Sample:
        conversation, values, entities = insert_pii(draft_factory(scenario), scenario)
        return Sample(
            conversation=conversation,
            labels=Annotation(
                id=scenario.id,
                scenario=scenario,
                pii_values=values,
                pii_entities=entities,
                provenance=Provenance(
                    model_id="ollama:qwen3:8b",
                    model_revision="test",
                    prompt_version="synthetic/v1",
                    prompt_sha256="a" * 64,
                    environment_sha256="b" * 64,
                    temperature=0.4,
                    attempt=1,
                    input_tokens=100,
                    output_tokens=20,
                    cost_usd=0,
                    latency_ms=1,
                    pii_scan="patterns",
                ),
            ),
        )

    return factory


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    yield exporter
    provider.shutdown()
