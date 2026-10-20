import argparse
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from convintel.errors import ConvIntelError
from convintel.telemetry import traced

from .judge import reviewable
from .schema import LABELS, VERSION, Annotation, Record, ReviewEvent, Sample, targets_of
from .storage import (
    append_jsonl,
    digest,
    file_hash,
    load_samples,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .taxonomy import INTENTS
from .validation import ensure_unique, validate_sample


class GoldenManifest(Record):
    version: str = "golden-v1"
    size: int
    conversations_sha256: str
    labels_sha256: str
    review_sha256: str
    by_intent: dict[str, int]
    by_channel: dict[str, int]


def decision(sample: Sample, events: list[ReviewEvent]) -> ReviewEvent | None:
    sha = digest(sample)
    matches = [
        event
        for event in events
        if event.id == sample.conversation.id
        and event.sample_sha256 == sha
        and event.action in {"approve", "reject"}
    ]
    return matches[-1] if matches else None


def show_sample(sample: Sample) -> str:
    scenario = sample.labels.scenario
    lines = [
        f"CANDIDATE {scenario.id} | sha256={digest(sample)}",
        f"intent={scenario.intent} channel={scenario.channel} resolved={scenario.resolved}",
        f"escalation_requested={scenario.escalation_requested}",
        f"PII scan={sample.labels.provenance.pii_scan}; manual PII and semantic review required",
    ]
    if isinstance(sample.labels, Annotation):
        target = sample.labels.scenario
        lines.append(f"customer_sentiment={target.start_sentiment} -> {target.end_sentiment}")
        lines.append(f"intent_description={target.intent_description}")
        if sample.labels.judge is not None:
            judge = sample.labels.judge
            targets = targets_of(target)
            lines.append(f"Blind judge {judge.model_id}, {judge.prompt_version}:")
            for label in LABELS:
                seen = getattr(judge.observation, label)
                status = "MISMATCH" if label in judge.mismatched_labels else "ok"
                lines.append(
                    f"  {label}: target={getattr(targets, label)} observed={seen.value} "
                    f"evidence={list(seen.evidence_turns)} {status}"
                )
    else:
        lines.append("ARCHIVED: no judge verdict; not eligible for human review or golden")
    for turn in sample.conversation.turns:
        lines.append(f"\n[{turn.index}] {turn.speaker.value}: {turn.text}")
        for span in sample.labels.pii_entities:
            if span.turn_index == turn.index:
                value = next(v for v in sample.labels.pii_values if v.slot.token == span.slot_token)
                lines.append(
                    f"  PII {span.entity_type} [{span.start}:{span.end}] "
                    f"surface={value.surface_form!r}; canonical={value.canonical_value!r}; "
                    f"spoken={value.spoken_form!r}"
                )
    return "\n".join(lines)


def review_batch(
    samples: list[Sample],
    ledger: Path,
    reviewer: str,
    batch_size: int = 10,
    *,
    ask: Callable[[str], str] = input,
    display: Callable[[str], None] = print,
    revisit: bool = False,
) -> int:
    if batch_size < 1 or not reviewer.strip():
        raise ConvIntelError("A reviewer and positive batch size are required")
    events = read_jsonl(ledger, ReviewEvent)
    count = 0
    for sample in samples:
        if not reviewable(sample):
            display(f"Skipped {sample.conversation.id}: missing, failed or stale judge verdict")
            continue
        if not revisit and decision(sample, events) is not None:
            continue
        if count == batch_size:
            break
        validate_sample(sample)
        display(show_sample(sample))
        for event in events:
            if event.id == sample.conversation.id and event.sample_sha256 == digest(sample):
                display(f"Previous {event.action} by {event.reviewer}: {event.note}")
        while True:
            try:
                answer = ask("[a]pprove / [r]eject / [n]ote / [s]kip / [q]uit: ").strip().lower()
                if answer in {"q", "quit"}:
                    return count
                if answer in {"s", "skip"}:
                    break
                actions: dict[str, Literal["approve", "reject", "note"]] = {
                    "a": "approve",
                    "approve": "approve",
                    "r": "reject",
                    "reject": "reject",
                    "n": "note",
                    "note": "note",
                }
                if answer not in actions:
                    display("Choose approve, reject, note, skip or quit.")
                    continue
                action = actions[answer]
                note = ask("Note (required for reject/note, no real personal data): ").strip()
            except (EOFError, KeyboardInterrupt):
                return count
            if action != "approve" and not note:
                display("A reason is required.")
                continue
            event = ReviewEvent(
                id=sample.conversation.id,
                sample_sha256=digest(sample),
                reviewer=reviewer,
                action=action,
                note=note,
                at=datetime.now(UTC),
            )
            with traced("synthetic.review", version=VERSION) as span:
                span.set_attribute("conversation.id", sample.conversation.id)
                span.set_attribute("review.action", action)
                append_jsonl(ledger, event)
            events.append(event)
            break
        count += 1
    return count


def export_golden(samples: list[Sample], ledger: Path, destination: Path) -> GoldenManifest:
    events = read_jsonl(ledger, ReviewEvent)
    selected: list[Sample] = []
    approvals: list[ReviewEvent] = []
    for sample in samples:
        latest = decision(sample, events)
        if latest is not None and latest.action == "approve" and reviewable(sample):
            validate_sample(sample)
            ensure_unique(sample, selected)
            selected.append(sample)
            approvals.append(latest)
    counts = Counter(s.labels.scenario.intent for s in selected)
    channels = Counter(s.conversation.channel.value for s in selected)
    if (
        len(selected) != 150
        or len({s.conversation.id for s in selected}) != 150
        or set(counts) != set(INTENTS)
        or Counter(counts.values()) != {2: 73, 1: 4}
        or channels != {"chat": 100, "voice": 50}
    ):
        raise ConvIntelError("golden-v1 requires 150 human-approved samples with the agreed quotas")
    if destination.exists():
        raise ConvIntelError(
            "Golden output already exists; immutable releases must not be overwritten"
        )
    destination.mkdir(parents=True)
    write_jsonl(destination / "conversations.jsonl", (s.conversation for s in selected))
    write_jsonl(
        destination / "labels.jsonl",
        (s.labels.model_copy(update={"status": "approved"}) for s in selected),
    )
    write_jsonl(destination / "review.jsonl", approvals)
    result = GoldenManifest(
        size=len(selected),
        conversations_sha256=file_hash(destination / "conversations.jsonl"),
        labels_sha256=file_hash(destination / "labels.jsonl"),
        review_sha256=file_hash(destination / "review.jsonl"),
        by_intent=dict(counts),
        by_channel=dict(channels),
    )
    write_json(destination / "manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Human-only, resumable synthetic data review")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--review-jsonl", type=Path)
    parser.add_argument("--revisit", action="store_true")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    ledger = args.review_jsonl or args.dataset / "review.jsonl"
    try:
        samples = load_samples(args.dataset)
        if not samples:
            raise ConvIntelError("No candidates to review")
        if args.export:
            print(export_golden(samples, ledger, args.export).model_dump_json(indent=2))
        else:
            review_batch(samples, ledger, args.reviewer, args.batch_size, revisit=args.revisit)
    except ConvIntelError as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
