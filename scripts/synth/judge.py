import asyncio
import hashlib
from collections.abc import Sequence
from typing import Any, Literal, cast

from pydantic import create_model

from convintel.config import Settings
from convintel.errors import ConvIntelError
from convintel.llm import LLMClient, LLMResponse, build_chat_model, load_prompt, wrap_conversation
from convintel.llm.prompts import Prompt
from convintel.telemetry import traced

from .scenarios import INTENT_DESCRIPTIONS, INTENTS
from .schema import (
    VERSION,
    Annotation,
    DraftTurn,
    JudgeObservation,
    JudgeRecord,
    ObservedIntent,
    Record,
    RunSpec,
    Sample,
    compare,
    targets_of,
)
from .storage import digest
from .validation import Rejected, masked_turns

JUDGE_PROMPT_VERSION = 2
TAXONOMY = "\n".join(f"- {name}: {text}" for name, text in INTENT_DESCRIPTIONS.items())
TAXONOMY_SHA256 = hashlib.sha256(TAXONOMY.encode()).hexdigest()

_IntentName = cast(Any, Literal).__getitem__(INTENTS)
_ClosedIntent = create_model("ObservedIntent", __base__=ObservedIntent, value=(_IntentName, ...))
# The tool schema offers only the 77 class names, so an invented intent fails parsing instead of
# reaching the comparison as a plausible-looking label.
JudgeSchema: type[JudgeObservation] = create_model(
    "JudgeObservation", __base__=JudgeObservation, intent=(_ClosedIntent, ...)
)


class JudgeInput(Record):
    """Everything the blind judge sees; the targets are deliberately absent."""

    taxonomy_sha256: str
    turns: tuple[DraftTurn, ...]


def judge_input(sample: Sample) -> JudgeInput:
    if not isinstance(sample.labels, Annotation):
        raise ConvIntelError("Historical samples cannot receive implicit judge approval")
    return JudgeInput(
        taxonomy_sha256=TAXONOMY_SHA256,
        turns=tuple(
            DraftTurn(
                speaker="customer" if turn.speaker == "customer" else "agent",
                text=text,
            )
            for turn, text in zip(
                sample.conversation.turns, masked_turns(sample, typed=True), strict=True
            )
        ),
    )


def evidence_errors(observation: JudgeObservation, speakers: Sequence[str]) -> tuple[str, ...]:
    """Evidence must exist, and customer-only labels may cite customer turns only.

    Sentiment labels are defined on the first and last customer turn, so their evidence has to
    include that turn; escalation is a customer request, so a `false` cites the customer turns
    that were checked for one.
    """
    customer = [index for index, speaker in enumerate(speakers) if speaker == "customer"]
    errors: list[str] = []
    for name in type(observation).model_fields:
        turns = getattr(observation, name).evidence_turns
        if not turns:
            errors.append(f"{name}:empty")
        elif any(index >= len(speakers) for index in turns):
            errors.append(f"{name}:out_of_range")
        elif name in {"start_sentiment", "end_sentiment", "escalation_requested"} and any(
            speakers[index] != "customer" for index in turns
        ):
            errors.append(f"{name}:agent_turn")
        elif name == "start_sentiment" and customer[0] not in turns:
            errors.append(f"{name}:missing_first_customer_turn")
        elif name == "end_sentiment" and customer[-1] not in turns:
            errors.append(f"{name}:missing_last_customer_turn")
    return tuple(errors)


def reviewable(sample: Sample) -> bool:
    if not isinstance(sample.labels, Annotation) or sample.labels.judge is None:
        return False
    judge = sample.labels.judge
    observation = judge.observation
    speakers = [turn.speaker.value for turn in sample.conversation.turns]
    return (
        judge.passed
        and not compare(observation, targets_of(sample.labels.scenario))
        and not evidence_errors(observation, speakers)
        and judge.input_sha256 == digest(judge_input(sample))
    )


async def observe(
    client: LLMClient, prompt: Prompt, turns: Sequence[DraftTurn], *, conversation_id: str
) -> LLMResponse[JudgeObservation]:
    conversation = "\n".join(
        f"[{index}] {turn.speaker}: {turn.text}" for index, turn in enumerate(turns)
    )
    with traced("synthetic.judge", version=VERSION) as span:
        span.set_attribute("conversation.id", conversation_id)
        span.set_attribute("model.id", client.model_id)
        span.set_attribute("prompt.version", prompt.id)
        response = await client.structured(
            JudgeSchema,
            prompt,
            operator="synthetic_judge",
            taxonomy=TAXONOMY,
            conversation=wrap_conversation(conversation),
        )
    observation = JudgeObservation.model_validate(response.output.model_dump())
    return LLMResponse(
        output=observation,
        model_id=response.model_id,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
    )


async def judge_sample(client: LLMClient, sample: Sample, run: RunSpec) -> JudgeRecord:
    prompt = await asyncio.to_thread(load_prompt, "synthetic_judge", JUDGE_PROMPT_VERSION)
    if digest(prompt) != run.judge_prompt_sha256:
        raise ConvIntelError("Judge prompt changed during generation; start a new run")
    data = judge_input(sample)
    if not isinstance(sample.labels, Annotation):
        raise ConvIntelError("Historical samples cannot receive implicit judge approval")
    response = await observe(client, prompt, data.turns, conversation_id=sample.conversation.id)
    record = JudgeRecord(
        model_id=response.model_id,
        model_revision=run.judge_model_revision,
        prompt_version=prompt.id,
        prompt_sha256=run.judge_prompt_sha256,
        input_sha256=digest(data),
        temperature=prompt.temperature,
        reasoning=False if client.model_id.startswith("ollama:") else None,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        observation=response.output,
        mismatched_labels=compare(response.output, targets_of(sample.labels.scenario)),
        evidence_errors=evidence_errors(response.output, [t.speaker for t in data.turns]),
    )
    if record.evidence_errors:
        raise Rejected("invalid_evidence", category="judge", judge=record)
    if record.mismatched_labels:
        raise Rejected("label_mismatch", category="judge", judge=record)
    return record


def configured_client(settings: Settings, prompt_name: str, version: int) -> LLMClient:
    prompt = load_prompt(prompt_name, version)
    if settings.llm_provider != "ollama":
        return LLMClient(settings, timeout_s=120)
    model = build_chat_model(settings, temperature=prompt.temperature)
    if "reasoning" not in type(model).model_fields:
        raise ConvIntelError("The installed Ollama integration cannot disable reasoning")
    model = model.model_copy(update={"reasoning": False})
    return LLMClient(settings, chat_model=model, timeout_s=120)
