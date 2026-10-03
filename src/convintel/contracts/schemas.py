"""Core data contracts shared by every module.

OWNER: human. Agents must not edit this file (see AGENTS.md). Every lane codes against
these types so that lanes can be built in parallel and integrated without surprises.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- input


class Channel(StrEnum):
    VOICE = "voice"
    SMS = "sms"
    CHAT = "chat"
    WHATSAPP = "whatsapp"


class Speaker(StrEnum):
    CUSTOMER = "customer"
    AGENT = "agent"
    BOT = "bot"
    UNKNOWN = "unknown"


class Turn(BaseModel):
    """One utterance or message. For voice, produced by transcription + diarization."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    speaker: Speaker
    text: str
    start_s: float | None = Field(default=None, ge=0, description="Voice only")
    end_s: float | None = Field(default=None, ge=0, description="Voice only")
    asr_confidence: float | None = Field(default=None, ge=0, le=1)


class Conversation(BaseModel):
    """Normalized conversation, regardless of channel. Every operator's input."""

    model_config = ConfigDict(frozen=True)

    id: str
    channel: Channel
    started_at: datetime
    language: str = "en"
    turns: tuple[Turn, ...]
    redacted: bool = False
    audio_uri: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(f"{t.speaker.value}: {t.text}" for t in self.turns)


# -------------------------------------------------------------------------- output


class Sentiment(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class SentimentOutput(BaseModel):
    overall: Sentiment
    customer_trajectory: list[Sentiment] = Field(
        default_factory=list[Sentiment], description="Customer sentiment per turn, in order"
    )


class IntentOutput(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    source: str = Field(description="'classifier' or 'llm_verifier'")
    top_k: list[tuple[str, float]] = Field(default_factory=list[tuple[str, float]])


class SummaryOutput(BaseModel):
    summary: str = Field(max_length=800)
    reason_for_contact: str
    resolved: bool | None
    action_items: list[str] = Field(default_factory=list)
    escalation_risk: float = Field(ge=0, le=1)


class PIIEntity(BaseModel):
    entity_type: str
    turn_index: int
    start: int
    end: int
    score: float = Field(ge=0, le=1)


class PIIOutput(BaseModel):
    entities: list[PIIEntity]
    redacted_conversation: Conversation


T = TypeVar("T", bound=BaseModel)


class OperatorResult(BaseModel, Generic[T]):
    """Envelope every operator returns. The metadata feeds telemetry and cost dashboards."""

    operator: str
    operator_version: str
    output: T
    latency_ms: float = Field(ge=0)
    model_id: str | None = None
    prompt_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = Field(default=None, ge=0)


class ConversationInsights(BaseModel):
    """Final aggregated result for one conversation, returned by the API."""

    conversation_id: str
    pipeline_version: str
    sentiment: OperatorResult[SentimentOutput] | None = None
    intent: OperatorResult[IntentOutput] | None = None
    summary: OperatorResult[SummaryOutput] | None = None
    pii_entity_count: int = 0
    total_latency_ms: float = 0
    total_cost_usd: float = 0
    trace_id: str | None = None
    errors: list[str] = Field(default_factory=list)
