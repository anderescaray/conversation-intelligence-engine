import argparse
import asyncio
import os
import time
from collections import Counter
from pathlib import Path

from prometheus_client import REGISTRY

from convintel.config import Settings
from convintel.errors import ConvIntelError, LLMError
from convintel.llm import LLMClient, load_prompt
from convintel.llm.pricing import compute_cost
from convintel.telemetry import setup, traced

from .pii import insert_pii
from .review import decision
from .scenarios import scenarios
from .schema import (
    VERSION,
    Annotation,
    Attempt,
    Provenance,
    Report,
    ReviewEvent,
    RunSpec,
    Sample,
    Scenario,
    draft_schema,
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
    validate_draft,
    validate_sample,
)

ROOT = Path(__file__).resolve().parents[2]


async def generate_sample(
    client: LLMClient,
    scenario: Scenario,
    run: RunSpec,
    attempt: int,
    scanner: NameScanner | None = None,
) -> Sample:
    prompt = await asyncio.to_thread(load_prompt, "synthetic", 1)
    if digest(prompt) != run.prompt_sha256:
        raise ConvIntelError("Prompt changed during generation; start a new run")
    with traced("synthetic.generate", version=VERSION) as span:
        span.set_attribute("conversation.id", scenario.id)
        span.set_attribute("model.id", client.model_id)
        span.set_attribute("prompt.version", prompt.id)
        response = await client.structured(
            draft_schema(scenario),
            prompt,
            operator="synthetic",
            scenario=scenario.model_dump_json(),
        )
        with traced("synthetic.validate", version=VERSION):
            await asyncio.to_thread(validate_draft, response.output, scenario, scanner)
        with traced("synthetic.pii", version=VERSION):
            conversation, values, entities = await asyncio.to_thread(
                insert_pii, response.output, scenario
            )
        sample = Sample(
            conversation=conversation,
            labels=Annotation(
                id=scenario.id,
                scenario=scenario,
                pii_values=values,
                pii_entities=entities,
                provenance=Provenance(
                    model_id=response.model_id,
                    model_revision=run.model_revision,
                    prompt_version=prompt.id,
                    prompt_sha256=run.prompt_sha256,
                    environment_sha256=run.environment_sha256,
                    temperature=prompt.temperature,
                    attempt=attempt,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=response.cost_usd,
                    latency_ms=response.latency_ms,
                    pii_scan="patterns+ner" if scanner else "patterns",
                ),
            ),
        )
        await asyncio.to_thread(validate_sample, sample, scanner)
        return sample


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


def report(samples: list[Sample], attempts: list[Attempt], target: int) -> Report:
    accepted = [attempt for attempt in attempts if attempt.accepted]
    elapsed = sum(attempt.elapsed_s for attempt in attempts)
    return Report(
        target=target,
        candidates=len(samples),
        attempts=len(attempts),
        rejects=len(attempts) - len(accepted),
        reject_rate=(len(attempts) - len(accepted)) / len(attempts) if attempts else 0,
        elapsed_s=elapsed,
        mean_attempt_s=elapsed / len(attempts) if attempts else 0,
        mean_accepted_attempt_s=(sum(a.elapsed_s for a in accepted) / len(accepted))
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
            Counter(t.value for s in samples for t in s.labels.scenario.customer_trajectory)
        ),
        by_resolved=dict(Counter(str(s.labels.scenario.resolved).lower() for s in samples)),
        by_escalation=dict(
            Counter(str(s.labels.scenario.escalation_requested).lower() for s in samples)
        ),
        by_pii=dict(Counter(e.entity_type for s in samples for e in s.labels.pii_entities)),
        rejection_reasons=dict(Counter(a.reason for a in attempts if not a.accepted)),
        missing_indices=sorted(set(range(target)) - {s.labels.scenario.index for s in samples}),
    )


def prepare_run(
    directory: Path,
    run: RunSpec,
    resume: bool,
) -> tuple[list[Sample], list[Attempt]]:
    path = directory / "run.json"
    if directory.exists():
        if not resume or not path.is_file():
            raise ConvIntelError("Output exists; use --resume only for a matching generator run")
        if RunSpec.model_validate_json(path.read_text()) != run:
            raise ConvIntelError("Resume configuration differs from the recorded run")
    else:
        directory.mkdir(parents=True)
        write_json(path, run)
    return load_samples(directory), read_jsonl(directory / "attempts.jsonl", Attempt)


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
) -> Report:
    if not 1 <= n <= 150 or max_attempts < 1 or seed < 0:
        raise ConvIntelError("Need n in [1,150], positive max_attempts and nonnegative seed")
    prompt = await asyncio.to_thread(load_prompt, "synthetic", 1)
    run = RunSpec(
        seed=seed,
        model_id=client.model_id,
        model_revision=model_revision,
        prompt_sha256=digest(prompt),
        environment_sha256=await asyncio.to_thread(file_hash, ROOT / "uv.lock"),
        pii_scan="patterns+ner" if scanner else "patterns",
    )
    samples, attempts = await asyncio.to_thread(prepare_run, directory, run, resume)
    if any(sample.labels.scenario.index >= n for sample in samples):
        raise ConvIntelError("Cannot shrink an existing run")
    events = await asyncio.to_thread(
        read_jsonl, review_jsonl or directory / "review.jsonl", ReviewEvent
    )
    regenerate = regenerate or set()
    rejected = {
        sample.labels.scenario.index
        for sample in samples
        if (event := decision(sample, events)) is not None and event.action == "reject"
    }
    if regenerate and (not resume or not regenerate <= rejected):
        raise ConvIntelError("Regeneration requires --resume and explicitly human-rejected indices")
    existing = {sample.conversation.id for sample in samples}
    for scenario in scenarios(seed)[:n]:
        replacing = scenario.index in regenerate
        if scenario.id in existing and not replacing:
            continue
        already = sum(a.id == scenario.id for a in attempts)
        limit = already + max_attempts if replacing else max_attempts
        for attempt in range(already + 1, limit + 1):
            before = usage(client.model_id)
            start = time.perf_counter()
            candidate: Sample | None = None
            reason = "accepted"
            try:
                candidate = await generate_sample(client, scenario, run, attempt, scanner)
                ensure_unique(candidate, (s for s in samples if s.conversation.id != scenario.id))
            except (Rejected, LLMError) as exc:
                reason = str(exc) if isinstance(exc, Rejected) else "llm_error"
                candidate = None
            elapsed = time.perf_counter() - start
            after = usage(client.model_id)
            tokens_in, tokens_out = (after[i] - before[i] for i in (0, 1))
            record = Attempt(
                id=scenario.id,
                attempt=attempt,
                accepted=candidate is not None,
                reason=reason,
                elapsed_s=elapsed,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                cost_usd=None
                if reason == "llm_error" and settings.llm_provider != "ollama"
                else compute_cost(settings.llm_provider, settings.llm_model, tokens_in, tokens_out),
            )
            await asyncio.to_thread(append_jsonl, directory / "attempts.jsonl", record)
            attempts.append(record)
            if candidate is not None:
                await asyncio.to_thread(append_jsonl, directory / "candidates.jsonl", candidate)
                samples = [s for s in samples if s.conversation.id != scenario.id]
                samples.append(candidate)
            print(record.model_dump_json(), flush=True)
            if candidate is not None:
                break
    samples.sort(key=lambda sample: sample.labels.scenario.index)
    result = report(samples, attempts, n).model_copy(
        update={
            "human_approved": sum(
                (event := decision(sample, events)) is not None and event.action == "approve"
                for sample in samples
            )
        }
    )
    await asyncio.to_thread(save_exports, directory, samples, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate evaluation candidates, never training data"
    )
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provider", choices=("ollama", "anthropic"), default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--model-revision", default="unresolved")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--regenerate-index", type=int, action="append")
    parser.add_argument("--review-jsonl", type=Path)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--ner-model")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    os.environ["OPENINFERENCE_HIDE_INPUTS"] = "true"
    os.environ["OPENINFERENCE_HIDE_OUTPUTS"] = "true"
    settings = Settings(
        llm_provider=args.provider,
        llm_model=args.model or ("qwen3:8b" if args.provider == "ollama" else "claude-haiku-4-5"),
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
                LLMClient(settings, timeout_s=args.timeout),
                settings,
                seed=args.seed,
                n=args.n,
                model_revision=args.model_revision,
                resume=args.resume,
                max_attempts=args.max_attempts,
                scanner=scanner,
                regenerate=set(args.regenerate_index or []),
                review_jsonl=args.review_jsonl,
            )
        )
    except ConvIntelError as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(result.model_dump_json(indent=2))
    return 1 if result.missing_indices else 0


if __name__ == "__main__":
    raise SystemExit(main())
