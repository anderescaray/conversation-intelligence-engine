"""Predict, before re-judging, which admitted candidates the changed definitions should flip.

The candidates were admitted under judge v2 (taxonomy v1, no labeling guide). Since then
resolution became first-contact resolution, sentiment became tone rather than topic, courtesy
became neutral, mild negative emotion became explicitly negative, and 21 intent descriptions
changed. This free local check is recorded before any paid call so that the re-judge can be
compared with a prediction it could not influence. Keyword lexicons are crude on purpose: the
point is an order of magnitude per field, not a second judge. Only IDs, indices and counts are
written; conversation text never leaves the process.
"""

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from convintel.contracts import Speaker
from convintel.errors import ConvIntelError

from .schema import LABELS, Annotation, Label, Record, Sample
from .storage import load_samples, write_json
from .taxonomy import DESCRIPTIONS


def _lexicon(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


NEGATIVE_EMOTION = _lexicon(
    r"\bannoy",
    r"\bfrustrat",
    r"\bworr",
    r"\bconcern",
    r"\bupset",
    r"\bangry",
    r"\bfurious",
    r"\birritat",
    r"\bridiculous",
    r"\bunacceptable",
    r"\bstress",
    r"\bpanic",
    r"\bfed up",
    r"\bdisappoint",
    r"\bterrible",
    r"\bawful",
    r"\bnightmare",
    r"\bscared",
    r"\bnervous",
    r"\banxious",
    r"\bunhappy",
    r"\bnot happy",
    r"\bconfus",
)
POSITIVE_EMOTION = _lexicon(
    r"\brelie[fv]",
    r"\bperfect",
    r"\bappreciate",
    r"\bbrilliant",
    r"\bgreat\b",
    r"\bwonderful",
    r"\bfantastic",
    r"\bglad",
    r"\bhappy",
    r"\bamazing",
    r"\bexcellent",
    r"\bawesome",
    r"\blifesaver",
    r"\blove",
    r"\bso much",
    r"\breally helpful",
)
COMPLETED = _lexicon(
    r"\b(i've|i have|we've|we have|has been|have been|is now|are now|it's now)\s+(\w+\s+)?"
    r"(processed|ordered|reset|refunded|blocked|cancell?ed|issued|unblocked|released|credited|"
    r"reversed|updated|changed|frozen|replaced|activated|removed|approved|sent)\b",
    r"\bon its way\b",
)
DEFERRED = _lexicon(
    r"\binvestigat",
    r"\breview",
    r"\bdisputes?\b",
    r"\b(call|contact|email|get back to|update) you\b",
    r"\bescalat",
    r"\bteam will\b",
    r"\bfollow[- ]?up\b",
    r"\bpassed (it )?on\b",
)


class FlipPrediction(Record):
    id: str
    index: int
    fields: tuple[Label, ...]


class PredictionReport(Record):
    source: str
    total: int
    candidates_with_any_flip: int
    predicted_by_field: dict[str, int]
    intent_description_changed: int
    predictions: list[FlipPrediction]


def predicted_fields(
    *,
    intent: str,
    resolved: bool,
    start_sentiment: str,
    end_sentiment: str,
    turns: Sequence[tuple[Speaker, str]],
) -> tuple[Label, ...]:
    """Fields whose target the new definitions should contradict, in LABELS order."""
    customer = [text for speaker, text in turns if speaker is Speaker.CUSTOMER]
    agent = [text for speaker, text in turns if speaker is Speaker.AGENT]
    if not customer or not agent:
        raise ConvIntelError("A candidate needs customer and agent turns")
    closing = " ".join(agent[-2:])
    completed, deferred = bool(COMPLETED.search(closing)), bool(DEFERRED.search(closing))
    first, last = customer[0], customer[-1]
    flips: set[Label] = set()
    # Upper bound: a changed description may or may not move the judge's choice.
    if DESCRIPTIONS[1][intent] != DESCRIPTIONS[2][intent]:
        flips.add("intent")
    if (not resolved and completed and not deferred) or (resolved and deferred and not completed):
        flips.add("resolved")
    if sentiment_flips(start_sentiment, first, courtesy_only=False):
        flips.add("start_sentiment")
    if sentiment_flips(end_sentiment, last, courtesy_only=True):
        flips.add("end_sentiment")
    return tuple(label for label in LABELS if label in flips)


def sentiment_flips(target: str, text: str, *, courtesy_only: bool) -> bool:
    negative = bool(NEGATIVE_EMOTION.search(text))
    if target == "negative":
        return not negative
    if target == "neutral":
        return negative
    # A positive closing without satisfaction beyond courtesy is now neutral.
    return courtesy_only and not POSITIVE_EMOTION.search(text)


def predict(sample: Sample) -> FlipPrediction:
    labels = sample.labels
    if not isinstance(labels, Annotation):
        raise ConvIntelError("Only annotated candidates can be predicted")
    scenario = labels.scenario
    fields = predicted_fields(
        intent=scenario.intent,
        resolved=scenario.resolved,
        start_sentiment=scenario.start_sentiment.value,
        end_sentiment=scenario.end_sentiment.value,
        turns=[(turn.speaker, turn.text) for turn in sample.conversation.turns],
    )
    return FlipPrediction(id=sample.conversation.id, index=scenario.index, fields=fields)


def predict_source(source: Path, output: Path) -> PredictionReport:
    if output.exists():
        raise ConvIntelError("Output exists; predictions are recorded once, before re-judging")
    predictions = [predict(sample) for sample in load_samples(source)]
    report = PredictionReport(
        source=str(source),
        total=len(predictions),
        candidates_with_any_flip=sum(bool(p.fields) for p in predictions),
        predicted_by_field={label: sum(label in p.fields for p in predictions) for label in LABELS},
        intent_description_changed=sum("intent" in p.fields for p in predictions),
        predictions=predictions,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict re-judge flips locally, for free")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = predict_source(args.source, args.output)
    except ConvIntelError as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(report.model_dump_json(indent=2, exclude={"predictions"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
