import asyncio

from convintel.config import Settings
from convintel.errors import ConvIntelError
from convintel.llm import LLMClient, build_chat_model, load_prompt, wrap_conversation
from convintel.telemetry import traced

from .schema import (
    VERSION,
    Annotation,
    DraftTurn,
    JudgeRecord,
    JudgeVerdict,
    Record,
    RunSpec,
    Sample,
    Scenario,
)
from .storage import digest
from .validation import Rejected, masked_turns


class JudgeInput(Record):
    scenario: Scenario
    turns: tuple[DraftTurn, ...]


def judge_input(sample: Sample) -> JudgeInput:
    if not isinstance(sample.labels, Annotation):
        raise ConvIntelError("Historical samples cannot receive implicit judge approval")
    return JudgeInput(
        scenario=sample.labels.scenario,
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


def evidence_valid(verdict: JudgeVerdict, sample: Sample) -> bool:
    last_customer = len(sample.conversation.turns) - 2
    for name in type(verdict).model_fields:
        check = getattr(verdict, name)
        if any(index >= len(sample.conversation.turns) for index in check.evidence_turns):
            return False
        if check.matches and name in {"start_sentiment", "end_sentiment"}:
            expected = 0 if name == "start_sentiment" else last_customer
            if set(check.evidence_turns) != {expected}:
                return False
    return True


def reviewable(sample: Sample) -> bool:
    if not isinstance(sample.labels, Annotation) or sample.labels.judge is None:
        return False
    judge = sample.labels.judge
    return (
        judge.verdict.passed
        and evidence_valid(judge.verdict, sample)
        and judge.input_sha256 == digest(judge_input(sample))
    )


async def judge_sample(client: LLMClient, sample: Sample, run: RunSpec) -> JudgeRecord:
    prompt = await asyncio.to_thread(load_prompt, "synthetic_judge", 1)
    if digest(prompt) != run.judge_prompt_sha256:
        raise ConvIntelError("Judge prompt changed during generation; start a new run")
    data = judge_input(sample)
    conversation = "\n".join(
        f"[{index}] {turn.speaker}: {turn.text}" for index, turn in enumerate(data.turns)
    )
    with traced("synthetic.judge", version=VERSION) as span:
        span.set_attribute("conversation.id", sample.conversation.id)
        span.set_attribute("model.id", client.model_id)
        span.set_attribute("prompt.version", prompt.id)
        response = await client.structured(
            JudgeVerdict,
            prompt,
            operator="synthetic_judge",
            targets=data.scenario.model_dump_json(),
            conversation=wrap_conversation(conversation),
        )
        record = JudgeRecord(
            model_id=response.model_id,
            model_revision=run.judge_model_revision,
            prompt_version=prompt.id,
            prompt_sha256=run.judge_prompt_sha256,
            input_sha256=digest(data),
            temperature=prompt.temperature,
            reasoning=run.reasoning,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            verdict=response.output,
        )
        span.set_attribute("judge.passed", record.verdict.passed)
        if not evidence_valid(record.verdict, sample):
            raise Rejected("invalid_evidence", category="judge", judge=record)
        if not record.verdict.passed:
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
