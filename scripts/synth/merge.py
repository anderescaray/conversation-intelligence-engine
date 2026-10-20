"""Combine candidate folders (re-judged and regenerated) into one reviewable candidate set.

Sources are only read. Every merged candidate must still be judge-passed under its own judge
version, valid and non-duplicated; a scenario may appear in one source only. The report counts
coverage against the 150-slot plan so that unfilled slots and missing intents stay visible.
"""

import argparse
from collections import Counter
from pathlib import Path

from convintel.errors import ConvIntelError

from .judge import reviewable
from .scenarios import scenarios
from .schema import Annotation, Record, Sample, ScenarioV3
from .storage import load_samples, write_json, write_jsonl
from .taxonomy import INTENTS
from .validation import ensure_unique, validate_sample


class MergeReport(Record):
    sources: dict[str, int]
    size: int
    target: int
    unfilled_indices: list[int]
    intents_missing: list[str]
    by_channel: dict[str, int]
    by_intent_kind: dict[str, int]
    by_resolved: dict[str, int]
    by_escalation: dict[str, int]
    by_sentiment: dict[str, int]
    by_generator_prompt: dict[str, int]
    by_judge_prompt: dict[str, int]
    intent_counts: dict[str, int]


def merge(sources: list[Path], output: Path, *, seed: int = 42) -> MergeReport:
    if output.exists():
        raise ConvIntelError("Output exists; merged candidate sets are immutable")
    merged: dict[str, Sample] = {}
    counts: dict[str, int] = {}
    for source in sources:
        samples = load_samples(source)
        counts[str(source)] = len(samples)
        for sample in samples:
            if sample.conversation.id in merged:
                raise ConvIntelError(f"{sample.conversation.id} appears in more than one source")
            if not reviewable(sample):
                raise ConvIntelError(
                    f"{sample.conversation.id} is not a current judge-passed candidate"
                )
            validate_sample(sample)
            merged[sample.conversation.id] = sample
    ordered = sorted(merged.values(), key=lambda s: s.labels.scenario.index)
    for index, sample in enumerate(ordered):
        ensure_unique(sample, ordered[:index])
    plan = scenarios(seed)
    present = {sample.labels.scenario.index for sample in ordered}
    labels = [sample.labels.scenario for sample in ordered]
    intents = Counter(scenario.intent for scenario in labels)
    output.mkdir(parents=True)
    write_jsonl(output / "candidates.jsonl", ordered)
    write_jsonl(output / "conversations.jsonl", (s.conversation for s in ordered))
    write_jsonl(output / "labels.jsonl", (s.labels for s in ordered))
    result = MergeReport(
        sources=counts,
        size=len(ordered),
        target=len(plan),
        unfilled_indices=sorted(set(range(len(plan))) - present),
        intents_missing=sorted(set(INTENTS) - set(intents)),
        by_channel=dict(Counter(s.conversation.channel.value for s in ordered)),
        by_intent_kind=dict(Counter(s.intent_kind for s in labels if isinstance(s, ScenarioV3))),
        by_resolved=dict(Counter(str(s.resolved).lower() for s in labels)),
        by_escalation=dict(Counter(str(s.escalation_requested).lower() for s in labels)),
        by_sentiment=dict(
            Counter(
                f"{s.start_sentiment.value}->{s.end_sentiment.value}"
                for s in labels
                if isinstance(s, ScenarioV3)
            )
        ),
        by_generator_prompt=dict(Counter(s.labels.provenance.prompt_version for s in ordered)),
        by_judge_prompt=dict(
            Counter(
                s.labels.judge.prompt_version
                for s in ordered
                if isinstance(s.labels, Annotation) and s.labels.judge is not None
            )
        ),
        intent_counts=dict(sorted(intents.items())),
    )
    write_json(output / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge judge-passed candidate folders")
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = merge(args.source, args.output, seed=args.seed)
    except ConvIntelError as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
