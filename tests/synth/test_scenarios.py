from collections import Counter

from scripts.synth.scenarios import ENTITY_TYPES, scenarios
from scripts.synth.taxonomy import INTENT_DESCRIPTIONS, INTENTS


def test_reproducible_full_golden_plan() -> None:
    plan = scenarios(42)
    assert plan == scenarios(42)
    assert plan != scenarios(43)
    assert len(plan) == 150
    assert len({s.id for s in plan}) == 150
    assert Counter(s.channel for s in plan) == {"chat": 100, "voice": 50}
    counts = Counter(s.intent for s in plan)
    assert set(counts) == set(INTENTS)
    assert Counter(counts.values()) == {2: 73, 1: 4}
    assert {s.intent for s in plan if s.channel == "chat"} == set(INTENTS)
    assert any(s.escalation_requested for s in plan[:15])
    assert any(s.channel == "voice" and s.slots for s in plan[:10])
    assert all("customer_trajectory" not in s.model_dump() for s in plan)
    assert set(INTENT_DESCRIPTIONS) == set(INTENTS)
    assert all(s.intent_description == INTENT_DESCRIPTIONS[s.intent] for s in plan)
    assert "currenc" in INTENT_DESCRIPTIONS["exchange_via_app"]
    assert all("\n" not in description for description in INTENT_DESCRIPTIONS.values())


def test_sampling_rules_hold_across_seeds() -> None:
    for seed in range(20):
        plan = scenarios(seed)
        unresolved = [s for s in plan if not s.resolved]
        assert Counter(s.resolved for s in plan) == {True: 75, False: 75}
        assert sum(s.end_sentiment == "positive" for s in unresolved) == 3
        for s in plan:
            assert 6 <= 2 * s.customer_turns <= 8
            if s.escalation_requested:
                assert s.start_sentiment == "negative"
                if not s.resolved:
                    assert s.end_sentiment == "negative"


def test_pii_coverage_placement_and_negative_controls() -> None:
    plan = scenarios(42)
    assert sum(not s.slots for s in plan) == 30
    assert sum(s.escalation_requested for s in plan) == 30
    for channel in ("chat", "voice"):
        assert {slot.entity_type for s in plan if s.channel == channel for slot in s.slots} == set(
            ENTITY_TYPES
        )
    partials = 0
    for s in plan:
        seen = set()
        for slot in s.slots:
            assert slot.token not in seen
            seen.add(slot.token)
            assert slot.turn_index < 2 * s.customer_turns
            if slot.speaker == "customer":
                assert slot.exposure == "full"
                assert slot.turn_index % 2 == 0
            else:
                partials += 1
                assert slot.exposure == "last_four"
                assert slot.turn_index % 2 == 1
                source = next(p for p in s.slots if p.token == slot.source_token)
                assert source.turn_index < slot.turn_index
                assert source.speaker == "customer"
    assert partials > 0
