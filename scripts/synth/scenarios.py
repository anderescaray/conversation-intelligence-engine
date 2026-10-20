import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import cast

from convintel.contracts import Sentiment
from convintel.errors import ConvIntelError

from .plausibility import (
    ESCALATION_KINDS,
    INTENT_KIND,
    LAST_FOUR_PII,
    PII_FAMILY,
    RULES,
    IntentKind,
    PlanVersion,
    allowed_pii,
)
from .schema import EntityType, Placement, Record, Scenario, ScenarioV3
from .taxonomy import (
    DESCRIPTIONS,
    INTENT_DESCRIPTIONS_V1,
    INTENTS,
    TaxonomyVersion,
)

ENTITY_TYPES: tuple[EntityType, ...] = (
    "PERSON",
    "CREDIT_CARD",
    "IBAN_CODE",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "ACCOUNT_NUMBER",
)


def legacy_scenarios(seed: int) -> tuple[Scenario, ...]:
    """Scenario plan v2, used by generators 2.0 and 3.0; reproduces their stored candidates."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = random.Random(seed)
    intents = list(INTENTS)
    rng.shuffle(intents)
    chat = iter([*intents, *intents[:23]])
    voice = iter(intents[23:73])
    resolutions = [False, True] * 75
    rng.shuffle(resolutions)
    rare_positive = set(rng.sample([i for i in range(150) if not resolutions[i] and i % 5 != 1], 3))
    result: list[Scenario] = []
    for index in range(150):
        sample_seed = int.from_bytes(hashlib.sha256(f"{seed}:{index}".encode()).digest()[:8])
        local = random.Random(sample_seed)
        channel = "voice" if index % 3 == 2 else "chat"
        count = local.randint(3, 4)
        negative, neutral, positive = Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE
        escalation = index % 5 == 1
        start = (
            negative if escalation else local.choices((negative, neutral, positive), (5, 4, 1))[0]
        )
        if index in rare_positive:
            start, end = neutral, positive
        elif resolutions[index]:
            end = local.choices((positive, neutral), (4, 1))[0]
        else:
            end = negative if escalation else local.choices((negative, neutral), (7, 3))[0]
        kind_index = (index // 3 + 3) % 6 if channel == "voice" else index % 6
        kinds = (
            () if index % 5 == 0 else (ENTITY_TYPES[kind_index], ENTITY_TYPES[(kind_index + 1) % 6])
        )
        slots = [
            Placement(
                entity_type=kind,
                token=f"[[{kind}_1]]",
                turn_index=2 * position,
                speaker="customer",
            )
            for position, kind in enumerate(kinds)
        ]
        eligible = [
            p
            for p in slots
            if p.entity_type in {"CREDIT_CARD", "IBAN_CODE", "PHONE_NUMBER", "ACCOUNT_NUMBER"}
        ]
        if eligible and index % 4 == 1:
            source = local.choice(eligible)
            slots.append(
                Placement(
                    entity_type=source.entity_type,
                    token=f"[[{source.entity_type}_LAST_FOUR_1]]",
                    turn_index=source.turn_index + 1,
                    speaker="agent",
                    exposure="last_four",
                    source_token=source.token,
                )
            )
        intent = next(voice if channel == "voice" else chat)
        result.append(
            Scenario(
                id=f"synth-v2-{seed}-{index:03d}",
                index=index,
                seed=seed,
                sample_seed=sample_seed,
                intent=intent,
                intent_description=INTENT_DESCRIPTIONS_V1[intent],
                channel=channel,
                customer_turns=count,
                start_sentiment=start,
                end_sentiment=end,
                resolved=resolutions[index],
                escalation_requested=escalation,
                persona=local.choice(
                    ("new to mobile banking", "experienced user", "busy customer")
                ),
                verbosity=local.choice(("brief", "moderate")),
                noise=local.choice(("none", "light")),
                slots=tuple(sorted(slots, key=lambda slot: slot.turn_index)),
            )
        )
    return tuple(result)


PERSONAS = ("new to mobile banking", "experienced user", "busy customer")
MAX_REROLLS = 2


def sample_seed(seed: int, index: int, reroll: int = 0) -> int:
    key = f"{seed}:{index}" if reroll == 0 else f"{seed}:{index}:reroll{reroll}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8])


def free_attributes(seed: int) -> dict[str, object]:
    """Attributes a same-stratum reroll may change; labels and PII plan are never among them."""
    local = random.Random(seed)
    return {
        "customer_turns": local.randint(3, 4),
        "persona": local.choice(PERSONAS),
        "verbosity": local.choice(("brief", "moderate")),
        "noise": local.choice(("none", "light")),
    }


def _weighted(rng: random.Random, weights: dict[Sentiment, int]) -> Sentiment:
    options = list(weights)
    return rng.choices(options, [weights[option] for option in options])[0]


def _assign_pii(intents: list[str], rng: random.Random) -> dict[int, list[EntityType]]:
    """Two plausible full PII types per non-control scenario, balancing counts across types.

    Scenarios with the fewest plausible types choose first, so constrained intents are not left
    with only over-represented types; ties are broken by the seeded generator.
    """
    indices = [index for index in range(len(intents)) if index % 5 != 0]
    keys = {index: rng.random() for index in indices}
    counts: Counter[str] = Counter()
    chosen: dict[int, list[EntityType]] = {}
    for index in sorted(indices, key=lambda i: (len(allowed_pii(intents[i])), keys[i])):
        options = list(allowed_pii(intents[index]))
        rng.shuffle(options)
        options.sort(key=lambda name: counts[name])
        pair: list[EntityType] = [cast(EntityType, name) for name in options[:2]]
        rng.shuffle(pair)
        counts.update(pair)
        chosen[index] = pair
    return chosen


def scenarios(
    seed: int, plan: PlanVersion = 4, taxonomy: TaxonomyVersion = 2
) -> tuple[ScenarioV3, ...]:
    """Plans v3/v4 for golden-v1: v2 channel and intent quotas with plausible labels and PII.

    Plan 4 is current; plan 3 reproduces the generator 3.1 pilot. Taxonomy 1 reproduces the
    descriptions used up to generator 3.2; taxonomy 2 (generator 3.3) changes no label.
    """
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = random.Random(seed)
    intents = list(INTENTS)
    rng.shuffle(intents)
    chat = iter([*intents, *intents[:23]])
    voice = iter(intents[23:73])
    channels = ["voice" if index % 3 == 2 else "chat" for index in range(150)]
    assigned = [next(voice if channel == "voice" else chat) for channel in channels]
    kinds: list[IntentKind] = [INTENT_KIND[intent] for intent in assigned]
    rules = RULES[plan]
    if plan == 3:
        resolutions = [False, True] * 75
        rng.shuffle(resolutions)
    else:
        # Kinds that are always answered take part of the resolved quota; the 75 unresolved
        # targets fall on the remaining intents, keeping 75/75 overall.
        fixed = [kinds[i] in rules.always_resolved_kinds for i in range(150)]
        free = [True] * (75 - sum(fixed)) + [False] * 75
        rng.shuffle(free)
        remaining = iter(free)
        resolutions = [True if fixed[i] else next(remaining) for i in range(150)]
    escalated: set[int] = set()
    for channel, count in (("chat", 20), ("voice", 10)):
        eligible = [
            i for i in range(150) if channels[i] == channel and kinds[i] in ESCALATION_KINDS
        ]
        escalated.update(rng.sample(eligible, count))
    rare_positive = set(
        rng.sample(
            [
                i
                for i in range(150)
                if not resolutions[i]
                and i not in escalated
                and kinds[i] in rules.rare_positive_kinds
            ],
            3,
        )
    )
    pii = _assign_pii(assigned, rng)
    confirmable = sorted(i for i, kinds_ in pii.items() if LAST_FOUR_PII & set(kinds_))
    partial = {
        index: rng.choice([kind for kind in pii[index] if kind in LAST_FOUR_PII])
        for index in sorted(rng.sample(confirmable, 30))
    }
    result: list[ScenarioV3] = []
    for index in range(150):
        intent, resolved = assigned[index], resolutions[index]
        kind: IntentKind = kinds[index]
        labels = random.Random(f"{seed}:{index}:labels")
        if index in escalated:
            start = Sentiment.NEGATIVE
            end = _weighted(labels, rules.end[kind, True]) if resolved else Sentiment.NEGATIVE
        elif index in rare_positive:
            start, end = Sentiment.NEUTRAL, Sentiment.POSITIVE
        else:
            start = _weighted(labels, rules.start[kind])
            end = _weighted(labels, rules.end[kind, resolved])
        slots = [
            Placement(
                entity_type=entity,
                token=f"[[{entity}_1]]",
                turn_index=2 * position,
                speaker="customer",
            )
            for position, entity in enumerate(pii.get(index, []))
        ]
        if index in partial:
            source = next(slot for slot in slots if slot.entity_type == partial[index])
            slots.append(
                Placement(
                    entity_type=source.entity_type,
                    token=f"[[{source.entity_type}_LAST_FOUR_1]]",
                    turn_index=source.turn_index + 1,
                    speaker="agent",
                    exposure="last_four",
                    source_token=source.token,
                )
            )
        seed_value = sample_seed(seed, index)
        result.append(
            ScenarioV3.model_validate(
                {
                    "id": f"synth-v{plan}-{seed}-{index:03d}",
                    "plan_version": plan,
                    "index": index,
                    "seed": seed,
                    "sample_seed": seed_value,
                    "intent": intent,
                    "intent_description": DESCRIPTIONS[taxonomy][intent],
                    "taxonomy_version": taxonomy,
                    "intent_kind": kind,
                    "pii_family": PII_FAMILY[intent],
                    "channel": channels[index],
                    "start_sentiment": start,
                    "end_sentiment": end,
                    "resolved": resolved,
                    "escalation_requested": index in escalated,
                    "slots": tuple(sorted(slots, key=lambda slot: slot.turn_index)),
                    **free_attributes(seed_value),
                }
            )
        )
    return tuple(result)


def reroll(scenario: ScenarioV3, attempt: int) -> ScenarioV3:
    """Same-stratum replacement: identical labels, intent, channel and PII plan."""
    if not 0 <= attempt <= MAX_REROLLS:
        raise ValueError(f"reroll must be within [0, {MAX_REROLLS}]")
    seed_value = sample_seed(scenario.seed, scenario.index, attempt)
    return ScenarioV3.model_validate(
        {
            **scenario.model_dump(),
            **free_attributes(seed_value),
            "sample_seed": seed_value,
            "reroll": attempt,
        }
    )


class PlanReport(Record):
    plan: str
    seed: int
    size: int
    by_channel: dict[str, int]
    by_intent_kind: dict[str, int]
    by_pii_family: dict[str, int]
    intents_covered: int
    no_pii_controls: int
    full_pii_by_type: dict[str, int]
    full_pii_by_type_and_channel: dict[str, dict[str, int]]
    last_four_by_type: dict[str, int]
    by_resolved: dict[str, int]
    escalation_by_kind: dict[str, int]
    escalation_by_channel: dict[str, int]
    start_by_kind: dict[str, dict[str, int]]
    end_by_kind_and_resolved: dict[str, dict[str, int]]
    sentiment_pairs: dict[str, int]
    rare_positive_unresolved: int
    customer_turns: dict[str, int]


def plan_report(plan: tuple[ScenarioV3, ...]) -> PlanReport:
    def grouped(key: str, values: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
        table: dict[str, Counter[str]] = {}
        for group, value in values:
            table.setdefault(group, Counter())[value] += 1
        return {group: dict(sorted(counter.items())) for group, counter in sorted(table.items())}

    full = [(s, slot) for s in plan for slot in s.slots if slot.exposure == "full"]
    return PlanReport(
        plan=plan[0].plan_prefix(),
        seed=plan[0].seed,
        size=len(plan),
        by_channel=dict(Counter(s.channel for s in plan)),
        by_intent_kind=dict(Counter(s.intent_kind for s in plan)),
        by_pii_family=dict(Counter(s.pii_family for s in plan)),
        intents_covered=len({s.intent for s in plan}),
        no_pii_controls=sum(not s.slots for s in plan),
        full_pii_by_type=dict(Counter(slot.entity_type for _, slot in full)),
        full_pii_by_type_and_channel=grouped(
            "type", [(slot.entity_type, s.channel) for s, slot in full]
        ),
        last_four_by_type=dict(
            Counter(slot.entity_type for s in plan for slot in s.slots if slot.exposure != "full")
        ),
        by_resolved=dict(Counter(str(s.resolved).lower() for s in plan)),
        escalation_by_kind=dict(Counter(s.intent_kind for s in plan if s.escalation_requested)),
        escalation_by_channel=dict(Counter(s.channel for s in plan if s.escalation_requested)),
        start_by_kind=grouped("kind", [(s.intent_kind, s.start_sentiment.value) for s in plan]),
        end_by_kind_and_resolved=grouped(
            "kind",
            [(f"{s.intent_kind}:resolved={s.resolved}", s.end_sentiment.value) for s in plan],
        ),
        sentiment_pairs=dict(
            Counter(f"{s.start_sentiment.value}->{s.end_sentiment.value}" for s in plan)
        ),
        rare_positive_unresolved=sum(
            not s.resolved and not s.escalation_requested and s.end_sentiment == "positive"
            for s in plan
        ),
        customer_turns=dict(Counter(str(s.customer_turns) for s in plan)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the current scenario plan report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(1, "Output exists; plan reports are immutable\n")
    try:
        report = plan_report(scenarios(args.seed))
    except (ValueError, ConvIntelError) as exc:
        parser.exit(1, f"{exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
