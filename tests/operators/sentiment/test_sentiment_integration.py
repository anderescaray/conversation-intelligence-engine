"""Real model: needs the `ml` group and the pinned revision in the local HF cache.

Provision it once with the command in the sentiment README. No test here uses the network.
"""

import socket
import ssl
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from sentiment_fakes import make_conversation

from convintel.contracts import Sentiment, Speaker
from convintel.operators.sentiment import SentimentOperator, model

AUTO_CONVERSION_JOIN_TIMEOUT_S = 10.0
"""Bounded wait for transformers' background hub thread, so the suite can never hang."""

pytestmark = pytest.mark.integration

C, A = Speaker.CUSTOMER, Speaker.AGENT


async def test_real_model_tracks_a_recovering_customer() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    conversation = make_conversation(
        (C, "This is the third time my internet has dropped today and I am furious."),
        (A, "I'm sorry about that. Let me check the line."),
        (C, "Fine. The router lights are blinking orange."),
        (A, "I've pushed a fix to your router, can you check now?"),
        (C, "It works now, thank you so much, that's great!"),
    )

    result = await SentimentOperator().run(conversation)

    trajectory = result.output.customer_trajectory
    assert len(trajectory) == 3
    assert trajectory[0] is Sentiment.NEGATIVE
    assert trajectory[-1] is Sentiment.POSITIVE
    assert result.output.overall is Sentiment.POSITIVE
    assert result.model_id == "cardiffnlp/twitter-roberta-base-sentiment-latest"


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Refuse and record every DNS lookup, and every connect, send or receive on an IP socket.

    I/O is guarded too, not only connects: an HTTP client may reuse a pooled connection opened
    by an earlier test, which needs neither a lookup nor a connect, and a blocking read on it
    would hang the suite. Local sockets (asyncio's self-pipe is a Unix socketpair) pass
    through, or the event loop would never wake up.
    """
    attempts: list[str] = []

    def refuse_lookup(*args: Any, **kwargs: Any) -> Any:
        attempts.append(f"getaddrinfo {args[0] if args else ''}")
        raise OSError("network disabled by test")

    def guard(cls: type[socket.socket], method: str) -> None:
        original = getattr(cls, method)

        def guarded(self: socket.socket, *args: Any, **kwargs: Any) -> Any:
            if self.family in (socket.AF_INET, socket.AF_INET6):
                attempts.append(f"{cls.__name__}.{method}")
                raise OSError("network disabled by test")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(cls, method, guarded)

    monkeypatch.setattr(socket, "getaddrinfo", refuse_lookup)
    io = ("connect", "connect_ex", "send", "sendall", "recv", "recv_into")
    for method in io:
        guard(socket.socket, method)
    for method in (*io, "read", "write"):
        guard(ssl.SSLSocket, method)
    yield attempts


async def test_warmup_opens_no_network_connection(no_network: list[str]) -> None:
    """Startup proof without HF_HUB_OFFLINE: offline mode would hide the hub calls that
    transformers makes despite `local_files_only=True` (safetensors auto-conversion)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model._cached_model.cache_clear()  # pyright: ignore[reportPrivateUsage]
    try:
        await SentimentOperator().warmup()
    finally:
        model._cached_model.cache_clear()  # pyright: ignore[reportPrivateUsage]

    # Wait for transformers' hub thread if one was started, so its calls are recorded.
    for thread in threading.enumerate():
        if thread.name.startswith("Thread-auto_conversion"):
            thread.join(timeout=AUTO_CONVERSION_JOIN_TIMEOUT_S)
            assert not thread.is_alive(), "auto-conversion thread still running after timeout"
    assert no_network == []
