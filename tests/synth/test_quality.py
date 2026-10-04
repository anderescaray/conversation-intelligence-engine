from collections.abc import Callable

import pytest
from pydantic import ValidationError

from scripts.synth.pii import values_for
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import DraftTurn, Scenario
from scripts.synth.validation import Rejected, validate_draft, validate_sample


@pytest.mark.parametrize("destination", [1, 4])
def test_placeholder_cannot_move_turn_or_speaker(draft_factory: Callable, destination: int) -> None:
    scenario = scenarios(42)[1]
    draft = draft_factory(scenario)
    slot = scenario.slots[0]
    draft.turns[slot.turn_index] = draft.turns[slot.turn_index].model_copy(
        update={"text": draft.turns[slot.turn_index].text.replace(slot.token, "")}
    )
    draft.turns[destination] = draft.turns[destination].model_copy(
        update={"text": draft.turns[destination].text + " " + slot.token}
    )
    with pytest.raises(Rejected, match="pii_placement"):
        validate_draft(draft, scenario)


def test_full_pii_cannot_be_planned_for_agent() -> None:
    data = scenarios(42)[1].model_dump()
    data["slots"][0].update({"speaker": "agent", "turn_index": 1})
    with pytest.raises(ValidationError, match="customer only"):
        Scenario.model_validate(data)


def test_partial_confirmations_are_derived_not_independently_sampled() -> None:
    for scenario in scenarios(42):
        values = {value.slot.token: value for value in values_for(scenario)}
        for slot in scenario.slots:
            if slot.source_token is not None:
                assert (
                    values[slot.token].canonical_value
                    == values[slot.source_token].canonical_value[-4:]
                )
                assert len(values[slot.token].canonical_value) == 4


def test_agent_cannot_repeat_full_name_without_placeholder(sample_factory: Callable) -> None:
    scenario = next(s for s in scenarios(42) if any(p.entity_type == "PERSON" for p in s.slots))
    sample = sample_factory(scenario)
    name = next(v for v in sample.labels.pii_values if v.slot.entity_type == "PERSON")
    turns = list(sample.conversation.turns)
    turns[1] = turns[1].model_copy(update={"text": turns[1].text + f" Hello, {name.surface_form}."})
    changed = sample.model_copy(
        update={"conversation": sample.conversation.model_copy(update={"turns": tuple(turns)})}
    )
    with pytest.raises(Rejected, match="agent_full_pii"):
        validate_sample(changed)


@pytest.mark.parametrize("block", [False, True])
def test_repetition_is_rejected_before_judging(draft_factory: Callable, block: bool) -> None:
    scenario = scenarios(42)[0]
    draft = draft_factory(scenario)
    if block:
        draft.turns[-2:] = draft.turns[:2]
    else:
        draft.turns[2] = DraftTurn(speaker="customer", text=draft.turns[0].text.upper() + "  ")
    with pytest.raises(Rejected, match="repeated_block" if block else "repeated_turn"):
        validate_draft(draft, scenario)


def test_turn_bounds_are_rechecked_after_deserialization(draft_factory: Callable) -> None:
    scenario = scenarios(42)[0]
    draft = draft_factory(scenario).model_copy(update={"turns": []})
    with pytest.raises(Rejected, match="turn_count"):
        validate_draft(draft, scenario)


def test_intent_label_cannot_leak_into_dialogue(draft_factory: Callable) -> None:
    scenario = scenarios(42)[0]
    draft = draft_factory(scenario)
    draft.turns[0] = DraftTurn(speaker="customer", text=f"My issue is {scenario.intent}.")
    with pytest.raises(Rejected, match="intent_label_leak"):
        validate_draft(draft, scenario)
