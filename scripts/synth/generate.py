import argparse
import asyncio
import json
import os
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, cast

from prometheus_client import REGISTRY
from pydantic import BaseModel, Field, ValidationError

from convintel.config import LLMProvider, Settings
from convintel.errors import ConvIntelError, LLMError
from convintel.llm import LLMClient, load_prompt
from convintel.llm.pricing import compute_cost
from convintel.telemetry import setup, traced

from .judge import JUDGE_PROMPT_VERSION, configured_client, judge_sample, reviewable
from .pii import insert_pii
from .review import decision
from .scenarios import scenarios
from .schema import (
    VERSION,
    Annotation,
    Attempt,
    GenerationProvenance,
    JudgeRecord,
    PhaseUsage,
    Report,
    ReviewEvent,
    RunSpec,
    Sample,
    Scenario,
    SkeletonDraft,
    add_usage,
    skeleton,
    skeleton_draft_schema,
)
from .storage import (
    append_jsonl,
    digest,
    file_hash,
    load_samples,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .validation import (
    NameScanner,
    Rejected,
    ensure_unique,
    local_name_scanner,
    skeleton_errors,
    validate_draft,
    validate_sample,
)

ROOT = Path(__file__).resolve().parents[2]
PROMPT_VERSION = 3


class Progress(BaseModel):
    phase: Literal["generation", "validation", "judge"] = "generation"
    judge: JudgeRecord | None = None
    repaired: bool = False
    usage: dict[str, PhaseUsage] = Field(default_factory=dict[str, PhaseUsage])


@contextmanager
def metered(progress: Progress, phase: str, client: LLMClient) -> Iterator[None]:
    """Attribute time, tokens and cost to one phase, including calls that later fail.

    Usage is read from the shared token counters, so a repair or a rejected answer still counts.
    An interrupted paid call may have unreported usage, so its cost becomes unknown, not zero.
    """
    before = usage(client.model_id)
    start = time.perf_counter()
    complete = True
    try:
        yield
    except BaseException as exc:
        complete = isinstance(exc, Rejected)
        raise
    finally:
        after = usage(client.model_id)
        delta_in, delta_out = after[0] - before[0], after[1] - before[1]
        provider, model = client.model_id.split(":", 1)
        cost = compute_cost(cast(LLMProvider, provider), model, delta_in, delta_out)
        if not complete and provider != "ollama":
            cost = None
        progress.usage[phase] = add_usage(
            progress.usage.get(phase, PhaseUsage()),
            PhaseUsage(
                seconds=time.perf_counter() - start,
                input_tokens=delta_in,
                output_tokens=delta_out,
                cost_usd=cost,
            ),
        )


def repair_block(previous: SkeletonDraft, errors: Sequence[str]) -> str:
    """Data for the single repair call; how to use it is stated in the versioned prompt."""
    listed = "\n".join(f"- {error}" for error in errors)
    return (
        f"<repair>\n<errors>\n{listed}\n</errors>\n"
        f"<previous_answer>\n{previous.model_dump_json()}\n</previous_answer>\n</repair>"
    )


async def generate_sample(
    client: LLMClient,
    scenario: Scenario,
    run: RunSpec,
    attempt: int,
    scanner: NameScanner | None = None,
    *,
    judge_client: LLMClient | None = None,
    previous: Sequence[Sample] = (),
    progress: Progress | None = None,
) -> Sample:
    progress = progress or Progress()
    prompt = await asyncio.to_thread(load_prompt, "synthetic", PROMPT_VERSION)
    if digest(prompt) != run.prompt_sha256:
        raise ConvIntelError("Prompt changed during generation; start a new run")
    plan = skeleton(scenario)
    schema = skeleton_draft_schema(scenario)
    variables = {
        "scenario": scenario.model_dump_json(),
        "skeleton": json.dumps([turn.model_dump() for turn in plan]),
    }
    with traced("synthetic.generate", version=VERSION) as span:
        span.set_attribute("conversation.id", scenario.id)
        span.set_attribute("model.id", client.model_id)
        span.set_attribute("prompt.version", prompt.id)
        with metered(progress, "generation", client):
            response = await client.structured(
                schema, prompt, operator="synthetic", feedback="", **variables
            )
            errors = skeleton_errors(response.output, plan)
            if errors:
                progress.repaired = True
                with traced("synthetic.repair", version=VERSION) as repair_span:
                    repair_span.set_attribute("conversation.id", scenario.id)
                    repair_span.set_attribute("repair.errors", len(errors))
                    response = await client.structured(
                        schema,
                        prompt,
                        operator="synthetic",
                        feedback=repair_block(response.output, errors),
                        **variables,
                    )
                    repair_span.set_attribute(
                        "repair.fixed", not skeleton_errors(response.output, plan)
                    )
                if skeleton_errors(response.output, plan):
                    raise Rejected("skeleton_mismatch")
        draft = response.output.as_draft()
        progress.phase = "validation"
        with traced("synthetic.validate", version=VERSION):
            await asyncio.to_thread(validate_draft, draft, scenario, scanner)
        with traced("synthetic.pii", version=VERSION):
            conversation, values, entities = await asyncio.to_thread(insert_pii, draft, scenario)
        generation = progress.usage["generation"]
        labels = Annotation(
            id=scenario.id,
            scenario=scenario,
            pii_values=values,
            pii_entities=entities,
            provenance=GenerationProvenance(
                model_id=response.model_id,
                model_revision=run.model_revision,
                prompt_version=prompt.id,
                prompt_sha256=run.prompt_sha256,
                environment_sha256=run.environment_sha256,
                temperature=prompt.temperature,
                attempt=attempt,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                cost_usd=generation.cost_usd,
                latency_ms=generation.seconds * 1000,
                pii_scan="patterns+ner" if scanner else "patterns",
                reasoning=run.reasoning,
                repaired=progress.repaired,
            ),
        )
        sample = Sample(conversation=conversation, labels=labels)
        with traced("synthetic.validate", version=VERSION):
            await asyncio.to_thread(validate_sample, sample, scanner)
            ensure_unique(sample, previous)
        progress.phase = "judge"
        judge = judge_client or client
        with metered(progress, "judge", judge):
            verdict = await judge_sample(judge, sample, run)
        progress.judge = verdict
        return Sample(
            conversation=conversation, labels=labels.model_copy(update={"judge": verdict})
        )


def usage(model_id: str) -> tuple[int, int]:
    counts = [
        int(
            REGISTRY.get_sample_value(
                "convintel_llm_tokens_total", {"model": model_id, "direction": direction}
            )
            or 0
        )
        for direction in ("in", "out")
    ]
    return counts[0], counts[1]


def scaled(total: PhaseUsage, count: int) -> PhaseUsage:
    return PhaseUsage(
        seconds=total.seconds / count,
        input_tokens=round(total.input_tokens / count),
        output_tokens=round(total.output_tokens / count),
        cost_usd=None if total.cost_usd is None else total.cost_usd / count,
    )


def report(
    samples: list[Sample], attempts: list[Attempt], target: int, run: RunSpec | None = None
) -> Report:
    accepted = [a for a in attempts if a.accepted]
    rejected = [a for a in attempts if not a.accepted]
    elapsed = sum(a.elapsed_s for a in attempts)
    categories = Counter(a.category for a in rejected)
    sample_times: dict[str, float] = {}
    for attempt in attempts:
        sample_times[attempt.id] = sample_times.get(attempt.id, 0) + attempt.elapsed_s
    phases = {
        "generation": [a.generation_usage for a in attempts if a.generation_usage],
        "judge": [a.judge_usage for a in attempts if a.judge_usage],
    }
    totals = {phase: add_usage(*parts) for phase, parts in phases.items() if parts}
    scenarios_seen = len({a.id for a in attempts})
    return Report(
        model_id=run.model_id if run else None,
        judge_model_id=run.judge_model_id if run else None,
        target=target,
        candidates=len(samples),
        attempts=len(attempts),
        rejects=len(rejected),
        reject_rate=len(rejected) / len(attempts) if attempts else 0,
        elapsed_s=elapsed,
        mean_attempt_s=elapsed / len(attempts) if attempts else 0,
        mean_accepted_attempt_s=sum(a.elapsed_s for a in accepted) / len(accepted)
        if accepted
        else None,
        input_tokens=sum(a.input_tokens for a in attempts),
        output_tokens=sum(a.output_tokens for a in attempts),
        cost_usd=None
        if any(a.cost_usd is None for a in attempts)
        else sum(a.cost_usd or 0 for a in attempts),
        by_channel=dict(Counter(s.conversation.channel.value for s in samples)),
        by_intent=dict(Counter(s.labels.scenario.intent for s in samples)),
        by_sentiment=dict(
            Counter(
                f"{s.labels.scenario.start_sentiment}->{s.labels.scenario.end_sentiment}"
                for s in samples
                if isinstance(s.labels, Annotation)
            )
        ),
        by_resolved=dict(Counter(str(s.labels.scenario.resolved).lower() for s in samples)),
        by_escalation=dict(
            Counter(str(s.labels.scenario.escalation_requested).lower() for s in samples)
        ),
        by_pii=dict(Counter(e.entity_type for s in samples for e in s.labels.pii_entities)),
        rejection_reasons=dict(Counter(a.reason for a in rejected)),
        rejection_categories=dict(categories),
        rejection_rates={key: count / len(attempts) for key, count in categories.items()},
        judge_failed_labels=dict(
            Counter(label for a in rejected if a.judge for label in a.judge.failed_labels)
        ),
        sample_elapsed_s=sample_times,
        timed_out_ids=sorted({a.id for a in rejected if a.category == "timeout"}),
        missing_indices=sorted(set(range(target)) - {s.labels.scenario.index for s in samples}),
        repairs=sum(a.repaired for a in attempts),
        repairs_fixed=sum(a.repaired and a.phase != "generation" for a in attempts),
        phase_totals=totals,
        phase_mean_per_scenario={
            phase: scaled(total, scenarios_seen) for phase, total in totals.items()
        },
        phase_per_admitted={phase: scaled(total, len(samples)) for phase, total in totals.items()}
        if samples
        else {},
    )


def prepare_run(directory: Path, run: RunSpec, resume: bool) -> tuple[list[Sample], list[Attempt]]:
    path = directory / "run.json"
    if directory.exists():
        if not resume or not path.is_file():
            raise ConvIntelError("Output exists; use --resume only for a matching generator run")
        try:
            matching = RunSpec.model_validate_json(path.read_text()) == run
        except ValidationError:
            matching = False
        if not matching:
            raise ConvIntelError("Resume configuration differs from the recorded run")
    else:
        directory.mkdir(parents=True)
        write_json(path, run)
    samples = load_samples(directory)
    if any(not reviewable(s) for s in samples):
        raise ConvIntelError("Current runs may contain only judge-passed, unchanged candidates")
    return samples, read_jsonl(directory / "attempts.jsonl", Attempt)


def save_exports(directory: Path, samples: list[Sample], result: Report) -> None:
    write_jsonl(directory / "conversations.jsonl", (sample.conversation for sample in samples))
    write_jsonl(directory / "labels.jsonl", (sample.labels for sample in samples))
    write_json(directory / "report.json", result)


async def generate_run(
    directory: Path,
    client: LLMClient,
    settings: Settings,
    *,
    seed: int,
    n: int,
    model_revision: str,
    resume: bool = False,
    max_attempts: int = 3,
    scanner: NameScanner | None = None,
    regenerate: set[int] | None = None,
    review_jsonl: Path | None = None,
    judge_client: LLMClient | None = None,
    sample_timeout_s: float = 120,
    reasoning: bool | None = None,
    judge_model_revision: str | None = None,
) -> Report:
    if not 1 <= n <= 150 or max_attempts < 1 or seed < 0 or not 0 < sample_timeout_s <= 120:
        raise ConvIntelError(
            "Need n in [1,150], positive attempts, nonnegative seed and timeout in (0,120]"
        )
    judge_client = judge_client or client
    prompt, judge_prompt = await asyncio.gather(
        asyncio.to_thread(load_prompt, "synthetic", PROMPT_VERSION),
        asyncio.to_thread(load_prompt, "synthetic_judge", JUDGE_PROMPT_VERSION),
    )
    run = RunSpec(
        seed=seed,
        model_id=client.model_id,
        model_revision=model_revision,
        prompt_sha256=digest(prompt),
        judge_model_id=judge_client.model_id,
        judge_model_revision=judge_model_revision
        or (model_revision if judge_client.model_id == client.model_id else "unresolved"),
        judge_prompt_sha256=digest(judge_prompt),
        environment_sha256=await asyncio.to_thread(file_hash, ROOT / "uv.lock"),
        pii_scan="patterns+ner" if scanner else "patterns",
        sample_timeout_s=sample_timeout_s,
        reasoning=reasoning,
    )
    samples, attempts = await asyncio.to_thread(prepare_run, directory, run, resume)
    if any(sample.labels.scenario.index >= n for sample in samples):
        raise ConvIntelError("Cannot shrink an existing run")
    events = await asyncio.to_thread(
        read_jsonl, review_jsonl or directory / "review.jsonl", ReviewEvent
    )
    regenerate = regenerate or set()
    rejected = {
        s.labels.scenario.index
        for s in samples
        if (event := decision(s, events)) is not None and event.action == "reject"
    }
    if regenerate and (not resume or not regenerate <= rejected):
        raise ConvIntelError("Regeneration requires --resume and explicitly human-rejected indices")
    existing = {sample.conversation.id for sample in samples}
    model_ids = {client.model_id, judge_client.model_id}
    for scenario in scenarios(seed)[:n]:
        replacing = scenario.index in regenerate
        previous_attempts = [a for a in attempts if a.id == scenario.id]
        if scenario.id in existing and not replacing:
            continue
        if not replacing and any(a.category == "timeout" for a in previous_attempts):
            continue
        consumed = 0 if replacing else sum(a.elapsed_s for a in previous_attempts)
        deadline = asyncio.get_running_loop().time() + max(0, sample_timeout_s - consumed)
        already = len(previous_attempts)
        limit = already + max_attempts if replacing else max_attempts
        for attempt in range(already + 1, limit + 1):
            before = {model: usage(model) for model in model_ids}
            start = time.perf_counter()
            candidate: Sample | None = None
            progress = Progress()
            reason = "accepted"
            category: Literal["accepted", "validator", "judge", "timeout", "llm"] = "accepted"
            try:
                async with asyncio.timeout_at(deadline):
                    candidate = await generate_sample(
                        client,
                        scenario,
                        run,
                        attempt,
                        scanner,
                        judge_client=judge_client,
                        previous=[s for s in samples if s.conversation.id != scenario.id],
                        progress=progress,
                    )
            except TimeoutError:
                candidate = None
                reason, category = "sample_timeout", "timeout"
            except Rejected as exc:
                candidate = None
                reason, category, progress.judge = str(exc), exc.category, exc.judge
            except LLMError as exc:
                candidate = None
                if isinstance(exc.__cause__, TimeoutError):
                    reason, category = "sample_timeout", "timeout"
                else:
                    reason, category = "llm_error", "llm"
            elapsed = time.perf_counter() - start
            tokens_in = tokens_out = 0
            costs: list[float | None] = []
            for model_id in model_ids:
                after = usage(model_id)
                delta_in, delta_out = (after[i] - before[model_id][i] for i in (0, 1))
                tokens_in += delta_in
                tokens_out += delta_out
                provider, model = model_id.split(":", 1)
                costs.append(compute_cost(cast(LLMProvider, provider), model, delta_in, delta_out))
            incomplete_paid = category in {"timeout", "llm"} and any(
                not m.startswith("ollama:") for m in model_ids
            )
            record = Attempt(
                id=scenario.id,
                attempt=attempt,
                accepted=candidate is not None,
                reason=reason,
                category=category,
                phase=progress.phase,
                judge=progress.judge,
                repaired=progress.repaired,
                generation_usage=progress.usage.get("generation"),
                judge_usage=progress.usage.get("judge"),
                elapsed_s=elapsed,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                cost_usd=None
                if incomplete_paid or any(c is None for c in costs)
                else sum(c or 0 for c in costs),
            )
            await asyncio.to_thread(append_jsonl, directory / "attempts.jsonl", record)
            attempts.append(record)
            if candidate is not None:
                await asyncio.to_thread(append_jsonl, directory / "candidates.jsonl", candidate)
                samples = [s for s in samples if s.conversation.id != scenario.id]
                samples.append(candidate)
            print(record.model_dump_json(), flush=True)
            if candidate is not None or category == "timeout":
                break
    samples.sort(key=lambda sample: sample.labels.scenario.index)
    result = report(samples, attempts, n, run).model_copy(
        update={
            "human_approved": sum(
                (event := decision(s, events)) is not None and event.action == "approve"
                for s in samples
            )
        }
    )
    await asyncio.to_thread(save_exports, directory, samples, result)
    return result


def default_model(provider: str) -> str:
    return "qwen3:8b" if provider == "ollama" else "claude-haiku-4-5"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate judge-passed evaluation candidates")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provider", choices=("ollama", "anthropic"), default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--model-revision", default="unresolved")
    parser.add_argument("--judge-provider", choices=("ollama", "anthropic"))
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-model-revision")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--regenerate-index", type=int, action="append")
    parser.add_argument("--review-jsonl", type=Path)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--ner-model")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    os.environ["OPENINFERENCE_HIDE_INPUTS"] = "true"
    os.environ["OPENINFERENCE_HIDE_OUTPUTS"] = "true"
    settings = Settings(
        llm_provider=args.provider, llm_model=args.model or default_model(args.provider)
    )
    judge_provider = args.judge_provider or args.provider
    judge_settings = Settings(
        llm_provider=judge_provider,
        llm_model=args.judge_model
        or (
            settings.llm_model if judge_provider == args.provider else default_model(judge_provider)
        ),
    )
    if args.trace:
        setup(settings=settings)
    destination = args.output or Path(f"data/generated/synthetic/run-{time.time_ns()}")
    print(f"Output directory: {destination}", flush=True)
    try:
        scanner = local_name_scanner(args.ner_model) if args.ner_model else None
        result = asyncio.run(
            generate_run(
                destination,
                configured_client(settings, "synthetic", PROMPT_VERSION),
                settings,
                judge_client=configured_client(
                    judge_settings, "synthetic_judge", JUDGE_PROMPT_VERSION
                ),
                judge_model_revision=args.judge_model_revision,
                seed=args.seed,
                n=args.n,
                model_revision=args.model_revision,
                resume=args.resume,
                max_attempts=args.max_attempts,
                scanner=scanner,
                regenerate=set(args.regenerate_index or []),
                review_jsonl=args.review_jsonl,
                sample_timeout_s=args.timeout,
                reasoning=False if settings.llm_provider == "ollama" else None,
            )
        )
    except ConvIntelError as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(result.model_dump_json(indent=2))
    return 1 if result.missing_indices else 0


if __name__ == "__main__":
    raise SystemExit(main())
