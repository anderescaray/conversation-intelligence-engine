from collections import Counter

from scripts.synth.scenarios import ENTITY_TYPES, INTENTS, scenarios


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
    assert any(s.escalation_requested for s in plan[:10])
    assert any(s.channel == "voice" and s.slots for s in plan[:10])
    assert all(len(s.customer_trajectory) == s.customer_turns for s in plan)


def test_pii_coverage_and_negative_controls() -> None:
    plan = scenarios(42)
    assert sum(not s.slots for s in plan) == 30
    assert Counter(s.resolved for s in plan) == {True: 75, False: 75}
    assert sum(s.escalation_requested for s in plan) == 30
    for channel in ("chat", "voice"):
        assert {slot.entity_type for s in plan if s.channel == channel for slot in s.slots} == set(
            ENTITY_TYPES
        )
    assert {slot.entity_type for s in plan for slot in s.slots} == {
        "PERSON",
        "CREDIT_CARD",
        "IBAN_CODE",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "ACCOUNT_NUMBER",
    }
