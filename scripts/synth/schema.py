from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from convintel.contracts import Conversation, PIIEntity, Sentiment

VERSION = "1.0.1"
EntityType = Literal[
    "PERSON", "CREDIT_CARD", "IBAN_CODE", "PHONE_NUMBER", "EMAIL_ADDRESS", "ACCOUNT_NUMBER"
]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Slot(Record):
    entity_type: EntityType
    token: str = Field(pattern=r"^\[\[[A-Z_]+_[0-9]+\]\]$")


class Scenario(Record):
    id: str = Field(pattern=r"^synth-v1-[0-9]+-[0-9]{3}$")
    index: int = Field(ge=0, lt=150)
    seed: int = Field(ge=0)
    sample_seed: int = Field(ge=0)
    intent: str
    channel: Literal["chat", "voice"]
    customer_turns: int = Field(ge=3, le=5)
    customer_trajectory: tuple[Sentiment, ...]
    resolved: bool
    escalation_requested: bool
    persona: str
    verbosity: Literal["brief", "moderate"]
    noise: Literal["none", "light"]
    slots: tuple[Slot, ...]


class DraftTurn(Record):
    speaker: Literal["customer", "agent"]
    text: str = Field(min_length=1, max_length=1000)


class Draft(Record):
    turns: list[DraftTurn] = Field(min_length=6, max_length=10)


def draft_schema(scenario: Scenario) -> type[Draft]:
    count = 2 * scenario.customer_turns
    return create_model(
        "Draft",
        __base__=Draft,
        turns=(list[DraftTurn], Field(min_length=count, max_length=count)),
    )


class PIIValue(Record):
    slot: Slot
    canonical_value: str
    spoken_form: str
    surface_form: str


class PIISpan(PIIEntity):
    model_config = ConfigDict(extra="forbid", frozen=True)
    slot_token: str
    score: float = Field(default=1.0, ge=1, le=1)


class Provenance(Record):
    generator_version: str = VERSION
    schema_version: str = "1"
    model_id: str
    model_revision: str
    prompt_version: str
    prompt_sha256: str
    environment_sha256: str
    temperature: float
    attempt: int = Field(ge=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float | None
    latency_ms: float = Field(ge=0)
    pii_scan: Literal["patterns", "patterns+ner"]


class Annotation(Record):
    id: str
    status: Literal["candidate", "approved"] = "candidate"
    scenario: Scenario
    pii_values: tuple[PIIValue, ...]
    pii_entities: tuple[PIISpan, ...]
    provenance: Provenance


class Sample(Record):
    conversation: Conversation
    labels: Annotation

    @model_validator(mode="after")
    def validate_alignment(self) -> "Sample":
        c, labels = self.conversation, self.labels
        s = labels.scenario
        if c.id != labels.id or c.id != s.id or c.channel != s.channel:
            raise ValueError("scenario identity mismatch")
        if c.redacted or c.language != "en" or c.started_at.utcoffset() is None:
            raise ValueError("invalid synthetic conversation metadata")
        if len(c.turns) != 2 * s.customer_turns:
            raise ValueError("turn count mismatch")
        if len(s.customer_trajectory) != s.customer_turns:
            raise ValueError("trajectory length mismatch")
        for index, turn in enumerate(c.turns):
            if turn.index != index or turn.speaker != ("customer" if index % 2 == 0 else "agent"):
                raise ValueError("invalid turn ordering")
        values = {v.slot.token: v for v in labels.pii_values}
        if len(values) != len(s.slots) or set(values) != {slot.token for slot in s.slots}:
            raise ValueError("PII inventory mismatch")
        seen: set[str] = set()
        ends: dict[int, int] = {}
        for span in labels.pii_entities:
            if not 0 <= span.turn_index < len(c.turns) or span.slot_token not in values:
                raise ValueError("invalid PII reference")
            value = values[span.slot_token]
            text = c.turns[span.turn_index].text
            if not ends.get(span.turn_index, 0) <= span.start < span.end <= len(text):
                raise ValueError("overlapping or invalid PII offsets")
            if text[span.start : span.end] != value.surface_form:
                raise ValueError("PII surface mismatch")
            if span.entity_type != value.slot.entity_type:
                raise ValueError("PII type mismatch")
            expected = value.spoken_form if s.channel == "voice" else value.canonical_value
            if value.surface_form != expected:
                raise ValueError("PII channel mismatch")
            ends[span.turn_index] = span.end
            seen.add(span.slot_token)
        if seen != set(values):
            raise ValueError("missing PII occurrences")
        return self


class Attempt(Record):
    id: str
    attempt: int
    accepted: bool
    reason: str
    elapsed_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


class RunSpec(Record):
    seed: int
    generator_version: str = VERSION
    model_id: str
    model_revision: str
    prompt_sha256: str
    environment_sha256: str
    pii_scan: str


class Report(Record):
    target: int
    candidates: int
    human_approved: int = 0
    attempts: int
    rejects: int
    reject_rate: float
    elapsed_s: float
    mean_attempt_s: float
    mean_accepted_attempt_s: float | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    by_channel: dict[str, int]
    by_intent: dict[str, int]
    by_sentiment: dict[str, int]
    by_resolved: dict[str, int]
    by_escalation: dict[str, int]
    by_pii: dict[str, int]
    rejection_reasons: dict[str, int]
    missing_indices: list[int]


class ReviewEvent(Record):
    id: str
    sample_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer: str = Field(min_length=1, max_length=80)
    action: Literal["approve", "reject", "note"]
    note: str = Field(default="", max_length=4000)
    at: datetime
