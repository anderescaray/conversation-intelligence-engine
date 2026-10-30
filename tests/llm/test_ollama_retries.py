"""Retry policy on the Ollama path, driven through the real ChatOllama/ollama/httpx stack.

A fake chat model would only prove `_is_retryable` accepts the exceptions we chose; serving
scripted HTTP through `httpx.MockTransport` proves the exceptions ollama actually raises (it
streams, so transport errors reach us as raw httpx exceptions) are the ones being retried.
"""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from convintel.config import Settings
from convintel.errors import LLMError
from convintel.llm import LLMClient
from convintel.llm.prompts import Prompt
from convintel.llm.testing import no_sleep

type Reply = httpx.Response | Exception


class Answer(BaseModel):
    label: str
    score: float


_FINAL = {
    "model": "llama3.1:8b",
    "created_at": "2026-01-01T00:00:00Z",
    "message": {"role": "assistant", "content": json.dumps({"label": "billing", "score": 0.5})},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 10,
    "eval_count": 5,
}


@pytest.fixture
def prompt() -> Prompt:
    return Prompt(
        name="demo",
        version=1,
        model="llama3.1:8b",
        temperature=0,
        system="Classify.",
        user_template="{conversation}",
    )


def _ok() -> httpx.Response:
    return httpx.Response(200, content=json.dumps(_FINAL).encode() + b"\n")


def _status(code: int) -> httpx.Response:
    return httpx.Response(code, json={"error": "scripted failure"})


class _DroppedStream(httpx.AsyncByteStream):
    """Sends one partial chunk, then the connection dies the way httpcore reports it."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        partial = {**_FINAL, "message": {"role": "assistant", "content": '{"lab'}, "done": False}
        yield json.dumps(partial).encode() + b"\n"
        raise httpx.RemoteProtocolError("peer closed connection without sending complete body")


def _dropped_mid_response() -> httpx.Response:
    return httpx.Response(200, stream=_DroppedStream())


def _client(replies: list[Reply], **kwargs: Any) -> tuple[LLMClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    settings = Settings(_env_file=None, llm_provider="ollama", llm_model="llama3.1:8b")  # pyright: ignore[reportCallIssue]
    model = ChatOllama(
        model=settings.llm_model,
        base_url="http://ollama.test",
        async_client_kwargs={"transport": httpx.MockTransport(handler)},
    )
    defaults: dict[str, Any] = {"sleep": no_sleep, "rand": lambda: 0.5}
    return LLMClient(settings, chat_model=model, **{**defaults, **kwargs}), requests


@pytest.mark.parametrize(
    "failure",
    [
        lambda: _status(500),
        lambda: _status(503),
        lambda: httpx.ConnectError("connection refused"),
        lambda: httpx.ReadError("connection reset by peer"),
        _dropped_mid_response,
    ],
    ids=["http-500", "http-503", "connect-error", "read-error", "dropped-mid-response"],
)
async def test_ollama_transient_errors_are_retried(
    prompt: Prompt, failure: Callable[[], Reply]
) -> None:
    client, requests = _client([failure(), _ok()])

    response = await client.structured(Answer, prompt, operator="demo", conversation="hi")

    assert response.output == Answer(label="billing", score=0.5)
    assert len(requests) == 2


@pytest.mark.parametrize("code", [400, 404])
async def test_ollama_client_errors_fail_fast(prompt: Prompt, code: int) -> None:
    client, requests = _client([_status(code), _ok()])

    with pytest.raises(LLMError, match="call failed: ResponseError"):
        await client.structured(Answer, prompt, operator="demo", conversation="hi")
    assert len(requests) == 1


async def test_ollama_retries_stop_at_the_limit(prompt: Prompt) -> None:
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    client, requests = _client(
        [_status(503), _dropped_mid_response(), _status(502), _ok()],
        max_retries=2,
        sleep=record,
        backoff_base_s=0.5,
    )

    with pytest.raises(LLMError, match="after 2 retries: ResponseError"):
        await client.structured(Answer, prompt, operator="demo", conversation="hi")
    assert len(requests) == 3
    assert delays == [0.5, 1.0]
