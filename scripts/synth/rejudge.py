"""Re-judge admitted candidates with the current judge, never touching their text or labels.

A candidate admitted under an earlier judge and taxonomy is only kept if the current blind judge
agrees with every target. The dialogue, PII and target labels are copied unchanged; only the
intent description moves to the taxonomy the judge was given, and the judge record is replaced.
Disagreements are reported per field so that the slot can be regenerated, never relabelled.
"""

import argparse
import asyncio
import os
import time
from collections import Counter
from pathlib import Path
from typing import Literal, cast

from convintel.config import LLMProvider, Settings
from convintel.errors import ConvIntelError, LLMError
from convintel.llm import LLMClient, load_prompt
from convintel.llm.pricing import compute_cost
from convintel.telemetry import traced

from .generate import default_model, usage
from .judge import JUDGE_PROMPT_VERSION, JUDGE_TAXONOMY, configured_client, judge_sample
from .schema import VERSION, Annotation, JudgeRecord, Record, RunSpec, Sample, ScenarioV3
from .storage import append_jsonl, digest, load_samples, read_jsonl, write_json, write_jsonl
from .taxonomy import DESCRIPTIONS
from .validation import Rejected


class RejudgeRun(Record):
    source: str
    source_run_sha256: str
    tool_version: str = VERSION
    judge_model_id: str
    judge_model_revision: str
    judge_prompt_sha256: str
    taxonomy_version: int


class RejudgeRecord(Record):
    id: str
    index: int
    kept: bool
    reason: str
    mismatched_labels: tuple[str, ...] = ()
    evidence_errors: tuple[str, ...] = ()
    judge: JudgeRecord | None = None
    elapsed_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


class RejudgeReport(Record):
    source: str
    judge_model_id: str
    judge_prompt_version: str
    taxonomy_version: int
    total: int
    judged: int
    kept: int
    flipped: int
    flipped_indices: list[int]
    flips_by_field: dict[str, int]
    flips_by_reason: dict[str, int]
    elapsed_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    stopped_reason: Literal["max_cost"] | None = None


def retaxonomised(sample: Sample) -> Sample:
    """The same candidate described with the judge's taxonomy; labels and text are unchanged."""
    labels = sample.labels
    if not isinstance(labels, Annotation) or not isinstance(labels.scenario, ScenarioV3):
        raise ConvIntelError("Only plan v3+ candidates can be re-judged")
    taxonomy = JUDGE_TAXONOMY[JUDGE_PROMPT_VERSION]
    scenario = ScenarioV3.model_validate(
        {
            **labels.scenario.model_dump(),
            "intent_description": DESCRIPTIONS[taxonomy][labels.scenario.intent],
            "taxonomy_version": taxonomy,
        }
    )
    return Sample(
        conversation=sample.conversation,
        labels=labels.model_copy(update={"scenario": scenario, "judge": None}),
    )


def prepare_output(output: Path, spec: RejudgeRun, resume: bool) -> None:
    if output.exists():
        recorded = output / "run.json"
        if not resume or RejudgeRun.model_validate_json(recorded.read_text()) != spec:
            raise ConvIntelError("Output exists; use --resume only for the same re-judge run")
        return
    output.mkdir(parents=True)
    write_json(output / "run.json", spec)


async def rejudge(
    source: Path,
    output: Path,
    client: LLMClient,
    *,
    judge_model_revision: str = "unresolved",
    max_cost_usd: float | None = None,
    resume: bool = False,
) -> RejudgeReport:
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise ConvIntelError("--max-cost-usd must be positive")
    prompt = await asyncio.to_thread(load_prompt, "synthetic_judge", JUDGE_PROMPT_VERSION)
    source_run = RunSpec.model_validate_json(
        await asyncio.to_thread((source / "run.json").read_text)
    )
    spec = RejudgeRun(
        source=str(source),
        source_run_sha256=digest(source_run),
        judge_model_id=client.model_id,
        judge_model_revision=judge_model_revision,
        judge_prompt_sha256=digest(prompt),
        taxonomy_version=JUDGE_TAXONOMY[JUDGE_PROMPT_VERSION],
    )
    await asyncio.to_thread(prepare_output, output, spec, resume)
    # judge_sample reads only the judge fields of the run specification.
    run = source_run.model_copy(
        update={
            "judge_model_id": client.model_id,
            "judge_model_revision": judge_model_revision,
            "judge_prompt_sha256": spec.judge_prompt_sha256,
        }
    )
    samples = await asyncio.to_thread(load_samples, source)
    records = await asyncio.to_thread(read_jsonl, output / "rejudge.jsonl", RejudgeRecord)
    done = {record.id for record in records}
    spent: float | None = 0.0
    stopped: Literal["max_cost"] | None = None
    with traced("synthetic.rejudge", version=VERSION) as span:
        span.set_attribute("model.id", client.model_id)
        for original in samples:
            if original.conversation.id in done:
                continue
            if max_cost_usd is not None and (spent is None or spent >= max_cost_usd):
                stopped = "max_cost"
                break
            sample = retaxonomised(original)
            before = usage(client.model_id)
            start = time.perf_counter()
            judge: JudgeRecord | None = None
            reason = "kept"
            try:
                judge = await judge_sample(client, sample, run)
            except Rejected as exc:
                reason = str(exc)
                judge = exc.judge if isinstance(exc.judge, JudgeRecord) else None
            except LLMError:
                reason = "llm_error"
            after = usage(client.model_id)
            tokens_in, tokens_out = after[0] - before[0], after[1] - before[1]
            provider, model = client.model_id.split(":", 1)
            cost = compute_cost(cast(LLMProvider, provider), model, tokens_in, tokens_out)
            if reason == "llm_error" and provider != "ollama":
                cost = None
            record = RejudgeRecord(
                id=sample.conversation.id,
                index=sample.labels.scenario.index,
                kept=reason == "kept",
                reason=reason,
                mismatched_labels=judge.mismatched_labels if judge else (),
                evidence_errors=judge.evidence_errors if judge else (),
                judge=judge,
                elapsed_s=time.perf_counter() - start,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                cost_usd=cost,
            )
            spent = None if spent is None or cost is None else spent + cost
            await asyncio.to_thread(append_jsonl, output / "rejudge.jsonl", record)
            if record.kept and judge is not None:
                kept = Sample(
                    conversation=sample.conversation,
                    labels=sample.labels.model_copy(update={"judge": judge}),
                )
                await asyncio.to_thread(append_jsonl, output / "candidates.jsonl", kept)
            records.append(record)
            print(
                f"{record.id} {record.reason} {list(record.mismatched_labels)} "
                f"{list(record.evidence_errors)}",
                flush=True,
            )
    kept_samples = await asyncio.to_thread(load_samples, output)
    flipped = [record for record in records if not record.kept]
    result = RejudgeReport(
        source=str(source),
        judge_model_id=client.model_id,
        judge_prompt_version=prompt.id,
        taxonomy_version=spec.taxonomy_version,
        total=len(samples),
        judged=len(records),
        kept=len(kept_samples),
        flipped=len(flipped),
        flipped_indices=sorted(record.index for record in flipped),
        flips_by_field=dict(
            Counter(label for record in flipped for label in record.mismatched_labels)
            + Counter("evidence" for record in flipped if record.evidence_errors)
        ),
        flips_by_reason=dict(Counter(record.reason for record in flipped)),
        elapsed_s=sum(record.elapsed_s for record in records),
        input_tokens=sum(record.input_tokens for record in records),
        output_tokens=sum(record.output_tokens for record in records),
        cost_usd=None
        if any(record.cost_usd is None for record in records)
        else sum(record.cost_usd or 0 for record in records),
        stopped_reason=stopped,
    )
    await asyncio.to_thread(
        write_jsonl, output / "conversations.jsonl", (s.conversation for s in kept_samples)
    )
    await asyncio.to_thread(write_jsonl, output / "labels.jsonl", (s.labels for s in kept_samples))
    await asyncio.to_thread(write_json, output / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-judge admitted candidates with the current judge"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("ollama", "anthropic"), default="anthropic")
    parser.add_argument("--model")
    parser.add_argument("--model-revision", default="unresolved")
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ["OPENINFERENCE_HIDE_INPUTS"] = "true"
    os.environ["OPENINFERENCE_HIDE_OUTPUTS"] = "true"
    settings = Settings(
        llm_provider=args.provider, llm_model=args.model or default_model(args.provider)
    )
    try:
        result = asyncio.run(
            rejudge(
                args.source,
                args.output,
                configured_client(settings, "synthetic_judge", JUDGE_PROMPT_VERSION),
                judge_model_revision=args.model_revision,
                max_cost_usd=args.max_cost_usd,
                resume=args.resume,
            )
        )
    except ConvIntelError as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(result.model_dump_json(indent=2))
    return 1 if result.stopped_reason else 0


if __name__ == "__main__":
    raise SystemExit(main())
