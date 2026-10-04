from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, create_model, model_validator

from convintel.contracts import Conversation, PIIEntity, Sentiment

VERSION = "2.0.0"
EntityType = Literal[
    "PERSON", "CREDIT_CARD", "IBAN_CODE", "PHONE_NUMBER", "EMAIL_ADDRESS", "ACCOUNT_NUMBER"
]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Slot(Record):
    entity_type: EntityType
    token: str = Field(pattern=r"^\[\[[A-Z_]+_[0-9]+\]\]$")


class Placement(Slot):
    turn_index: int = Field(ge=0, lt=10)
    speaker: Literal["customer", "agent"]
    exposure: Literal["full", "last_four"] = "full"
    source_token: str | None = None


class ScenarioData[S: Slot](Record):
    id: str = Field(pattern=r"^synth-v[12]-[0-9]+-[0-9]{3}$")
    index: int = Field(ge=0, lt=150)
    seed: int = Field(ge=0)
    sample_seed: int = Field(ge=0)
    intent: str
    channel: Literal["chat", "voice"]
    customer_turns: int = Field(ge=3, le=5)
    resolved: bool
    escalation_requested: bool
    persona: str
    verbosity: Literal["brief", "moderate"]
    noise: Literal["none", "light"]
    slots: tuple[S, ...]


class LegacyScenario(ScenarioData[Slot]):
    customer_trajectory: tuple[Sentiment, ...]


class Scenario(ScenarioData[Placement]):
    intent_description: str = Field(min_length=1)
    start_sentiment: Sentiment
    end_sentiment: Sentiment

    @model_validator(mode="after")
    def validate_plan(self) -> "Scenario":
        if self.id != f"synth-v2-{self.seed}-{self.index:03d}":
            raise ValueError("scenario version or identity mismatch")
        if self.escalation_requested and self.start_sentiment != Sentiment.NEGATIVE:
            raise ValueError("escalation must start negative")
        if (
            self.escalation_requested
            and not self.resolved
            and self.end_sentiment != Sentiment.NEGATIVE
        ):
            raise ValueError("unresolved escalation must end negative")
        slots = {slot.token: slot for slot in self.slots}
        if len(slots) != len(self.slots):
            raise ValueError("duplicate PII token")
        for slot in self.slots:
            if slot.turn_index >= self.customer_turns * 2:
                raise ValueError("PII turn out of range")
            expected = "customer" if slot.turn_index % 2 == 0 else "agent"
            if slot.speaker != expected:
                raise ValueError("PII speaker mismatch")
            if slot.exposure == "full":
                if slot.speaker != "customer" or slot.source_token is not None:
                    raise ValueError("full PII belongs to the customer only")
            else:
                source = slots.get(slot.source_token or "")
                if (
                    slot.speaker != "agent"
                    or source is None
                    or source.exposure != "full"
                    or source.turn_index >= slot.turn_index
                    or source.entity_type != slot.entity_type
                    or slot.entity_type
                    not in {"CREDIT_CARD", "IBAN_CODE", "PHONE_NUMBER", "ACCOUNT_NUMBER"}
                ):
                    raise ValueError("invalid partial confirmation")
        return self


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
    slot: Placement | Slot
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


class GenerationProvenance(Provenance):
    schema_version: str = "2"
    reasoning: bool | None = None


class LabelCheck(Record):
    matches: StrictBool
    evidence_turns: tuple[Annotated[int, Field(ge=0, lt=10)], ...] = Field(max_length=10)

    @model_validator(mode="after")
    def require_evidence(self) -> "LabelCheck":
        if self.matches and not self.evidence_turns:
            raise ValueError("matching labels require evidence")
        return self


class JudgeVerdict(Record):
    intent: LabelCheck
    resolved: LabelCheck
    escalation_requested: LabelCheck
    start_sentiment: LabelCheck
    end_sentiment: LabelCheck

    @property
    def failed_labels(self) -> tuple[str, ...]:
        return tuple(name for name in type(self).model_fields if not getattr(self, name).matches)

    @property
    def passed(self) -> bool:
        return not self.failed_labels


class JudgeRecord(Record):
    model_id: str
    model_revision: str
    prompt_version: str
    prompt_sha256: str
    input_sha256: str
    temperature: float
    reasoning: bool | None = None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float | None
    latency_ms: float = Field(ge=0)
    verdict: JudgeVerdict


class LegacyAnnotation(Record):
    id: str
    status: Literal["candidate", "approved"] = "candidate"
    scenario: LegacyScenario
    pii_values: tuple[PIIValue, ...]
    pii_entities: tuple[PIISpan, ...]
    provenance: Provenance


class Annotation(Record):
    id: str
    status: Literal["candidate", "approved"] = "candidate"
    scenario: Scenario
    pii_values: tuple[PIIValue, ...]
    pii_entities: tuple[PIISpan, ...]
    provenance: GenerationProvenance
    judge: JudgeRecord | None = None


class Sample(Record):
    conversation: Conversation
    labels: Annotation | LegacyAnnotation

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
        if isinstance(s, LegacyScenario) and len(s.customer_trajectory) != s.customer_turns:
            raise ValueError("trajectory length mismatch")
        for index, turn in enumerate(c.turns):
            if turn.index != index or turn.speaker != ("customer" if index % 2 == 0 else "agent"):
                raise ValueError("invalid turn ordering")
        values = {v.slot.token: v for v in labels.pii_values}
        if len(values) != len(s.slots) or set(values) != {slot.token for slot in s.slots}:
            raise ValueError("PII inventory mismatch")
        if any(values[slot.token].slot != slot for slot in s.slots):
            raise ValueError("PII plan mismatch")
        seen: list[str] = []
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
            if isinstance(value.slot, Placement) and span.turn_index != value.slot.turn_index:
                raise ValueError("PII placement mismatch")
            expected = value.spoken_form if s.channel == "voice" else value.canonical_value
            if value.surface_form != expected:
                raise ValueError("PII channel mismatch")
            ends[span.turn_index] = span.end
            seen.append(span.slot_token)
        if set(seen) != set(values) or (isinstance(s, Scenario) and len(seen) != len(values)):
            raise ValueError("missing or repeated PII occurrences")
        return self


class Attempt(Record):
    id: str
    attempt: int
    accepted: bool
    reason: str
    category: Literal["accepted", "validator", "judge", "timeout", "llm"] = "validator"
    phase: Literal["generation", "validation", "judge"] = "generation"
    judge: JudgeRecord | None = None
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
    judge_model_id: str
    judge_model_revision: str
    judge_prompt_sha256: str
    environment_sha256: str
    pii_scan: str
    sample_timeout_s: float = Field(default=120, gt=0, le=120)
    reasoning: bool | None = None


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
    rejection_categories: dict[str, int]
    rejection_rates: dict[str, float]
    judge_failed_labels: dict[str, int]
    sample_elapsed_s: dict[str, float]
    timed_out_ids: list[str]
    missing_indices: list[int]


class ReviewEvent(Record):
    id: str
    sample_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer: str = Field(min_length=1, max_length=80)
    action: Literal["approve", "reject", "note"]
    note: str = Field(default="", max_length=4000)
    at: datetime
