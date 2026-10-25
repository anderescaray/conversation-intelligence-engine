import asyncio
from typing import Any

import anthropic
import httpx2
import ollama
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from prometheus_client import REGISTRY
from pydantic import BaseModel, Field, SecretStr

from convintel.config import Settings
from convintel.errors import LLMError
from convintel.llm import LLMClient, build_chat_model, load_prompt
from convintel.llm.pricing import effective_temperature
from convintel.llm.prompts import Prompt
from convintel.llm.testing import FakeStructuredChatModel, no_sleep, tool_answer

_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


class Answer(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


_VALID = {"label": "billing", "score": 0.5}
_INVALID = {"label": "billing", "score": 7}


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"anthropic_api_key": SecretStr("test-key"), **overrides}
    return Settings(_env_file=None, **values)  # pyright: ignore[reportCallIssue]


@pytest.fixture
def prompt() -> Prompt:
    return Prompt(
        name="demo",
        version=1,
        model="claude-haiku-4-5",
        temperature=0,
        system="Classify.",
        user_template="{conversation}",
    )


def _no_jitter() -> float:
    """Jitter factor 0.5 + 0.5 = 1.0, so delays equal the plain exponential backoff."""
    return 0.5


def _client(responses: list[AIMessage | Exception], **kwargs: Any) -> LLMClient:
    model = FakeStructuredChatModel(responses=responses)
    defaults: dict[str, Any] = {"sleep": no_sleep, "rand": _no_jitter}
    return LLMClient(_settings(), chat_model=model, **{**defaults, **kwargs})


def _status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    return cls("provider error", response=httpx2.Response(status, request=_REQUEST), body=None)


def _fake(client: LLMClient) -> FakeStructuredChatModel:
    model = client._chat_model
    assert isinstance(model, FakeStructuredChatModel)
    return model


async def test_structured_returns_output_and_accounting(prompt: Prompt) -> None:
    client = _client([tool_answer(Answer, _VALID, input_tokens=100, output_tokens=20)])

    response = await client.structured(Answer, prompt, operator="demo", conversation="hi")

    assert response.output == Answer(label="billing", score=0.5)
    assert response.model_id == "anthropic:claude-haiku-4-5"
    assert (response.input_tokens, response.output_tokens) == (100, 20)
    assert response.cost_usd == pytest.approx((100 * 1.0 + 20 * 5.0) / 1_000_000)
    assert response.latency_ms >= 0


async def test_invalid_output_gets_one_repair_retry(prompt: Prompt) -> None:
    client = _client(
        [
            tool_answer(Answer, _INVALID, input_tokens=100, output_tokens=20),
            tool_answer(Answer, _VALID, input_tokens=150, output_tokens=25),
        ]
    )

    response = await client.structured(Answer, prompt, operator="demo", conversation="hi")

    assert response.output.score == 0.5
    # Both attempts are billed.
    assert (response.input_tokens, response.output_tokens) == (250, 45)
    calls = _fake(client).calls
    assert len(calls) == 2
    repair = str(calls[1][-1].content)
    assert "did not match the Answer schema" in repair
    assert '"score": 7' in repair


async def test_missing_tool_call_is_repaired_too(prompt: Prompt) -> None:
    client = _client([AIMessage(content="I think it is billing."), tool_answer(Answer, _VALID)])
    response = await client.structured(Answer, prompt, operator="demo", conversation="hi")
    assert response.output.label == "billing"
    assert "I think it is billing." in str(_fake(client).calls[1][-1].content)


async def test_invalid_output_twice_raises_llm_error(prompt: Prompt) -> None:
    client = _client([tool_answer(Answer, _INVALID), tool_answer(Answer, _INVALID)])
    with pytest.raises(LLMError, match="after one repair attempt"):
        await client.structured(Answer, prompt, operator="demo", conversation="hi")
    assert len(_fake(client).calls) == 2


async def test_transient_errors_retry_with_exponential_backoff(prompt: Prompt) -> None:
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    rate_limited = anthropic.RateLimitError(
        "rate limited", response=httpx2.Response(429, request=_REQUEST), body=None
    )
    client = _client(
        [anthropic.APITimeoutError(request=_REQUEST), rate_limited, tool_answer(Answer, _VALID)],
        sleep=record,
        backoff_base_s=0.5,
    )

    response = await client.structured(Answer, prompt, operator="demo", conversation="hi")

    assert response.output.label == "billing"
    assert delays == [0.5, 1.0]


@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (anthropic.InternalServerError, 500),
        (anthropic.APIStatusError, 502),
        (anthropic.ServiceUnavailableError, 503),
        (anthropic.OverloadedError, 529),
    ],
    ids=["500", "502", "503", "529-overloaded"],
)
async def test_anthropic_server_errors_are_retried(
    prompt: Prompt, cls: type[anthropic.APIStatusError], status: int
) -> None:
    client = _client([_status_error(cls, status), tool_answer(Answer, _VALID)])
    response = await client.structured(Answer, prompt, operator="demo", conversation="hi")
    assert response.output.label == "billing"
    assert len(_fake(client).calls) == 2


async def test_client_errors_other_than_429_are_not_retried(prompt: Prompt) -> None:
    client = _client([_status_error(anthropic.BadRequestError, 400), tool_answer(Answer, _VALID)])
    with pytest.raises(LLMError, match="call failed"):
        await client.structured(Answer, prompt, operator="demo", conversation="hi")
    assert len(_fake(client).calls) == 1


async def test_backoff_is_jittered_by_the_injected_random_source(prompt: Prompt) -> None:
    delays: list[float] = []
    draws = iter([0.0, 0.999])

    async def record(delay: float) -> None:
        delays.append(delay)

    client = _client(
        [TimeoutError(), TimeoutError(), tool_answer(Answer, _VALID)],
        sleep=record,
        rand=lambda: next(draws),
        backoff_base_s=0.5,
    )

    await client.structured(Answer, prompt, operator="demo", conversation="hi")

    # Factor is 0.5 + rand: attempt 0 -> 0.5 * 0.5, attempt 1 -> 1.0 * 1.499.
    assert delays == pytest.approx([0.25, 1.499])


async def test_retries_exhausted_raise_llm_error(prompt: Prompt) -> None:
    client = _client(
        [ConnectionError(), ConnectionError(), ollama.ResponseError("busy", 429)],
        max_retries=2,
    )
    with pytest.raises(LLMError, match="after 2 retries"):
        await client.structured(Answer, prompt, operator="demo", conversation="hi")


@pytest.mark.parametrize(
    "error",
    [
        anthropic.AuthenticationError(
            "bad key", response=httpx2.Response(401, request=_REQUEST), body=None
        ),
        ollama.ResponseError("model not found", 404),
    ],
)
async def test_non_transient_provider_errors_fail_fast(prompt: Prompt, error: Exception) -> None:
    client = _client([error, tool_answer(Answer, _VALID)])
    with pytest.raises(LLMError, match="call failed") as info:
        await client.structured(Answer, prompt, operator="demo", conversation="hi")
    assert info.value.__cause__ is error
    assert len(_fake(client).calls) == 1


async def test_bugs_are_not_wrapped(prompt: Prompt) -> None:
    client = _client([ValueError("bug")])
    with pytest.raises(ValueError, match="bug"):
        await client.structured(Answer, prompt, operator="demo", conversation="hi")


async def test_global_timeout_bounds_retries(prompt: Prompt) -> None:
    client = _client(
        [TimeoutError(), tool_answer(Answer, _VALID)],
        sleep=asyncio.sleep,
        backoff_base_s=10,
        timeout_s=0.05,
    )
    with pytest.raises(LLMError, match="timeout"):
        await client.structured(Answer, prompt, operator="demo", conversation="hi")


async def test_metrics_count_tokens_and_cost(prompt: Prompt) -> None:
    model_id = "anthropic:claude-haiku-4-5"

    def sample(name: str, labels: dict[str, str]) -> float:
        return REGISTRY.get_sample_value(name, labels) or 0.0

    tokens_in = {"model": model_id, "direction": "in"}
    tokens_out = {"model": model_id, "direction": "out"}
    cost = {"model": model_id, "operator": "metrics-test"}
    before = [
        sample("convintel_llm_tokens_total", tokens_in),
        sample("convintel_llm_tokens_total", tokens_out),
        sample("convintel_llm_cost_usd_total", cost),
    ]
    client = _client([tool_answer(Answer, _VALID, input_tokens=1000, output_tokens=100)])

    await client.structured(Answer, prompt, operator="metrics-test", conversation="hi")

    assert sample("convintel_llm_tokens_total", tokens_in) - before[0] == 1000
    assert sample("convintel_llm_tokens_total", tokens_out) - before[1] == 100
    assert sample("convintel_llm_cost_usd_total", cost) - before[2] == pytest.approx(0.0015)


async def test_failed_repair_still_counts_tokens_and_cost(prompt: Prompt) -> None:
    model_id = "anthropic:claude-haiku-4-5"
    cost = {"model": model_id, "operator": "failed-repair-test"}
    tokens_in = {"model": model_id, "direction": "in"}

    def sample(name: str, labels: dict[str, str]) -> float:
        return REGISTRY.get_sample_value(name, labels) or 0.0

    cost_before = sample("convintel_llm_cost_usd_total", cost)
    tokens_before = sample("convintel_llm_tokens_total", tokens_in)
    client = _client(
        [
            tool_answer(Answer, _INVALID, input_tokens=100, output_tokens=20),
            tool_answer(Answer, _INVALID, input_tokens=100, output_tokens=20),
        ]
    )

    with pytest.raises(LLMError, match="after one repair attempt"):
        await client.structured(Answer, prompt, operator="failed-repair-test", conversation="hi")

    # Both billed attempts are recorded even though the call failed.
    assert sample("convintel_llm_tokens_total", tokens_in) - tokens_before == 200
    assert sample("convintel_llm_cost_usd_total", cost) - cost_before == pytest.approx(
        2 * (100 * 1.0 + 20 * 5.0) / 1_000_000
    )


def test_missing_api_key_raises_clear_llm_error() -> None:
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY is not set"):
        build_chat_model(settings, temperature=0)


async def test_missing_api_key_surfaces_on_call(prompt: Prompt) -> None:
    client = LLMClient(Settings(_env_file=None))  # pyright: ignore[reportCallIssue]
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        await client.structured(Answer, prompt, operator="demo", conversation="hi")


@pytest.mark.parametrize("provider", ["google", "openai"])
def test_unwired_providers_raise_llm_error(provider: str) -> None:
    with pytest.raises(LLMError, match="not wired up"):
        build_chat_model(_settings(llm_provider=provider), temperature=0)


@pytest.mark.parametrize(
    ("env", "chat_class", "model_id", "expected_cost"),
    [
        (
            {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"},
            ChatAnthropic,
            "anthropic:claude-haiku-4-5",
            (100 * 1.0 + 20 * 5.0) / 1_000_000,
        ),
        (
            {
                "LLM_PROVIDER": "ollama",
                "LLM_MODEL": "llama3.1:8b",
                "OLLAMA_BASE_URL": "http://gpu-box:11434",
            },
            ChatOllama,
            "ollama:llama3.1:8b",
            0.0,
        ),
    ],
    ids=["anthropic", "ollama"],
)
async def test_switching_provider_is_config_only(
    monkeypatch: pytest.MonkeyPatch,
    prompt: Prompt,
    env: dict[str, str],
    chat_class: type[BaseChatModel],
    model_id: str,
    expected_cost: float,
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]

    # The real provider model is built from env alone, with no network I/O.
    chat = build_chat_model(settings, temperature=prompt.temperature)
    assert isinstance(chat, chat_class)
    assert getattr(chat, "model", None) == settings.llm_model
    if isinstance(chat, ChatOllama):
        assert chat.base_url == "http://gpu-box:11434"

    # The same code path then yields provider-specific accounting.
    fake = FakeStructuredChatModel(responses=[tool_answer(Answer, _VALID)])
    client = LLMClient(settings, chat_model=fake)
    response = await client.structured(Answer, prompt, operator="demo", conversation="hi")
    assert response.model_id == model_id
    assert response.cost_usd == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    ("model", "sent"), [("claude-sonnet-5-5", None), ("claude-haiku-4-5", 0.0)]
)
def test_temperature_is_omitted_only_for_models_that_reject_it(
    model: str, sent: float | None
) -> None:
    chat = build_chat_model(_settings(llm_provider="anthropic", llm_model=model), temperature=0)
    assert isinstance(chat, ChatAnthropic) and chat.model == model
    assert chat.temperature == sent
    assert effective_temperature("anthropic", model, 0) == sent
    # What reaches the API (langchain routes sampling parameters through extra_body).
    payload = chat._get_request_payload([HumanMessage("hi")])  # pyright: ignore[reportPrivateUsage]
    sent_body = payload.get("extra_body") or {}
    assert "temperature" not in payload
    assert sent_body.get("temperature") == sent


def test_summary_prompt_model_does_not_override_settings() -> None:
    settings = _settings(llm_provider="ollama", llm_model="llama3.1:8b")
    prompt = load_prompt("summary", 1)
    assert prompt.model == "claude-haiku-4-5"
    chat = build_chat_model(settings, temperature=prompt.temperature)
    assert isinstance(chat, ChatOllama)
    assert chat.model == "llama3.1:8b"
