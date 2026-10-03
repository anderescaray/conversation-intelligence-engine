"""Operator protocol. OWNER: human. Agents must not edit this file.

An operator is any unit that takes a Conversation and returns an OperatorResult.
The LangGraph pipeline (Phase 2) only knows about this protocol, so lanes can ship
operators independently.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from .schemas import Conversation, OperatorResult


@runtime_checkable
class Operator[T: BaseModel](Protocol):
    name: str
    """Stable identifier, used in metrics labels and traces (e.g. 'intent')."""

    version: str
    """Bump when behaviour changes (model, prompt, thresholds). Semver string."""

    async def run(self, conversation: Conversation) -> OperatorResult[T]:
        """Analyze a (redacted) conversation. Must not mutate the input."""
        ...


class Transcriber(Protocol):
    """Turns audio into a Conversation with speaker-labelled turns."""

    async def transcribe(self, audio_uri: str, conversation_id: str) -> Conversation: ...
