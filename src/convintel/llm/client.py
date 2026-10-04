"""The single path for every LLM call in convintel (AGENTS.md rule 4).

`LLMClient.structured()` takes a versioned prompt and a Pydantic schema and returns a validated
instance plus the accounting every `OperatorResult` needs (model id, tokens, cost, latency).
Provider and model come only from `Settings`, so switching Anthropic <-> Ollama is an env change.

Spans for the provider call come from OpenInference's LangChain auto-instrumentation (wired up
by the telemetry lane); this module adds the Prometheus counters and never logs prompt or
conversation text, only model, operator and token counts.
"""

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

import anthropic
import httpx
import ollama
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from convintel.config import Settings, get_settings
from convintel.errors import LLMError
from convintel.llm.metrics import LLM_COST_USD, LLM_TOKENS
from convintel.llm.pricing import compute_cost
from convintel.llm.prompts import Prompt

logger = logging.getLogger(__name__)

_MAX_REPAIR_DETAIL_CHARS = 2000

_PROVIDER_ERRORS = (anthropic.APIError, ollama.ResponseError, ollama.RequestError, httpx.HTTPError)
"""Expected provider failures, surfaced as LLMError. Anything else is a bug and propagates."""


class LLMResponse[T: BaseModel](BaseModel):
    """A validated structured output plus the accounting for the call that produced it."""

    output: T
    model_id: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)


def _is_retryable(exc: BaseException) -> bool:
    """Rate limits, server-side failures, timeouts and dropped connections are transient.

    Anthropic 5xx includes 529 `OverloadedError`, which is common under load and is not a
    subclass of `InternalServerError`, so the status code is checked instead of the class.
    """
    if isinstance(exc, ollama.ResponseError):
        return exc.status_code == 429
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return isinstance(
        exc,
        anthropic.RateLimitError
        | anthropic.APIConnectionError  # includes APITimeoutError
        | httpx.TimeoutException
        | ConnectionError
        | TimeoutError,
    )


def build_chat_model(settings: Settings, *, temperature: float) -> BaseChatModel:
    """Instantiate the configured provider's chat model. Construction does no network I/O."""
    provider, model = settings.llm_provider, settings.llm_model
    match provider:
        case "anthropic":
            if settings.anthropic_api_key is None:
                raise LLMError(
                    "ANTHROPIC_API_KEY is not set; it is required when LLM_PROVIDER=anthropic"
                )
            chat = init_chat_model(
                model,
                model_provider="anthropic",
                temperature=temperature,
                api_key=settings.anthropic_api_key.get_secret_value(),
                # LLMClient owns retries; SDK retries would multiply attempts and hide backoff.
                max_retries=0,
            )
        case "ollama":
            chat = init_chat_model(
                model,
                model_provider="ollama",
                temperature=temperature,
                base_url=settings.ollama_base_url,
            )
        case "google" | "openai":
            raise LLMError(
                f"LLM_PROVIDER={provider} is not wired up yet; use 'anthropic' or 'ollama'"
            )
    if not isinstance(chat, BaseChatModel):
        raise LLMError(f"init_chat_model returned an unexpected {type(chat).__name__}")
    return chat


class LLMClient:
    """Async structured-output client. One instance can be shared across operators.

    `chat_model` overrides the provider model (tests inject fakes). `sleep` and `rand` (a
    source of floats in [0, 1)) are injectable so backoff can be asserted deterministically.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        chat_model: BaseChatModel | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand: Callable[[], float] = random.random,
    ) -> None:
        self._settings = settings or get_settings()
        self._chat_model = chat_model
        self._models_by_temperature: dict[float, BaseChatModel] = {}
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._sleep = sleep
        self._rand = rand

    @property
    def model_id(self) -> str:
        return f"{self._settings.llm_provider}:{self._settings.llm_model}"

    def _model_for(self, temperature: float) -> BaseChatModel:
        if self._chat_model is not None:
            return self._chat_model
        if temperature not in self._models_by_temperature:
            self._models_by_temperature[temperature] = build_chat_model(
                self._settings, temperature=temperature
            )
        return self._models_by_temperature[temperature]

    async def structured[T: BaseModel](
        self, schema: type[T], prompt: Prompt, *, operator: str, **variables: str
    ) -> LLMResponse[T]:
        """Render `prompt` with `variables` and return the model's answer as a `schema` instance.

        Transient provider errors are retried with exponential backoff; an answer that does not
        fit the schema gets exactly one repair attempt. The whole call, retries included, is
        bounded by `timeout_s`. Every failure surfaces as `LLMError`.
        """
        messages = prompt.render(**variables)
        runnable = self._model_for(prompt.temperature).with_structured_output(
            schema, include_raw=True
        )
        tokens = [0, 0]
        cost: float | None = None
        start = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout_s):
                result = await self._invoke(runnable, messages, tokens)
                parsed = result.get("parsed")
                if not isinstance(parsed, schema):
                    logger.warning(
                        "invalid structured output, repairing (operator=%s model=%s)",
                        operator,
                        self.model_id,
                    )
                    repair = [*messages, _repair_message(result, schema)]
                    result = await self._invoke(runnable, repair, tokens)
                    parsed = result.get("parsed")
        except TimeoutError as exc:
            raise LLMError(f"LLM call exceeded the {self._timeout_s:g}s timeout") from exc
        finally:
            # Failed calls are billed too, so usage and cost are recorded on every exit path.
            cost = self._record_usage(operator, *tokens)

        if not isinstance(parsed, schema):
            raise LLMError(
                f"{self.model_id} returned output that does not fit {schema.__name__} "
                "after one repair attempt"
            )
        return LLMResponse(
            output=parsed,
            model_id=self.model_id,
            input_tokens=tokens[0],
            output_tokens=tokens[1],
            cost_usd=cost,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    async def _invoke(
        self, runnable: Runnable[Any, Any], messages: list[BaseMessage], tokens: list[int]
    ) -> dict[str, Any]:
        """One logical call with backoff retries; adds the call's usage to `tokens`."""
        attempt = 0
        while True:
            try:
                result = await runnable.ainvoke(messages)
                break
            except Exception as exc:
                if _is_retryable(exc) and attempt < self._max_retries:
                    # Jitter in [0.5x, 1.5x) spreads out clients that were rate limited together.
                    delay = self._backoff_base_s * 2**attempt * (0.5 + self._rand())
                    logger.warning(
                        "transient LLM error %s, retry %d/%d in %.2fs (model=%s)",
                        type(exc).__name__,
                        attempt + 1,
                        self._max_retries,
                        delay,
                        self.model_id,
                    )
                    attempt += 1
                    await self._sleep(delay)
                    continue
                if _is_retryable(exc):
                    raise LLMError(
                        f"{self.model_id} still failing after {self._max_retries} retries: "
                        f"{type(exc).__name__}"
                    ) from exc
                if isinstance(exc, _PROVIDER_ERRORS):
                    raise LLMError(f"{self.model_id} call failed: {type(exc).__name__}") from exc
                raise

        if not isinstance(result, dict):
            raise LLMError(f"unexpected structured-output result {type(result).__name__}")
        raw = result.get("raw")
        if isinstance(raw, AIMessage) and raw.usage_metadata is not None:
            tokens[0] += raw.usage_metadata["input_tokens"]
            tokens[1] += raw.usage_metadata["output_tokens"]
        return result

    def _record_usage(self, operator: str, input_tokens: int, output_tokens: int) -> float | None:
        """Increment token and cost counters; returns the cost (None if the model is unpriced)."""
        LLM_TOKENS.labels(model=self.model_id, direction="in").inc(input_tokens)
        LLM_TOKENS.labels(model=self.model_id, direction="out").inc(output_tokens)
        cost = compute_cost(
            self._settings.llm_provider, self._settings.llm_model, input_tokens, output_tokens
        )
        if cost is not None:
            LLM_COST_USD.labels(model=self.model_id, operator=operator).inc(cost)
        return cost


def _repair_message(result: dict[str, Any], schema: type[BaseModel]) -> HumanMessage:
    """Ask the model to fix its previous answer.

    The previous answer is quoted back as text instead of replaying the assistant message:
    providers like Anthropic reject a tool call that is not followed by a tool result.
    """
    raw = result.get("raw")
    previous: str = "(no answer)"
    if isinstance(raw, AIMessage):
        previous = json.dumps(raw.tool_calls[0]["args"]) if raw.tool_calls else str(raw.content)
    error = result.get("parsing_error") or "the answer did not call the required output schema"
    return HumanMessage(
        content=(
            f"Your previous answer did not match the {schema.__name__} schema.\n"
            f"Validation error: {str(error)[:_MAX_REPAIR_DETAIL_CHARS]}\n"
            f"Previous answer: {previous[:_MAX_REPAIR_DETAIL_CHARS]}\n"
            "Answer again using the schema exactly."
        )
    )
