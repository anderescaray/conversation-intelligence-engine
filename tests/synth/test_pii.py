from collections.abc import Callable

import pytest
from pydantic import ValidationError

from scripts.synth.pii import insert_pii, value_key, values_for
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import DraftTurn, Sample
from scripts.synth.validation import validate_sample


def test_all_pii_surfaces_are_reproducible(
    draft_factory: Callable, sample_factory: Callable
) -> None:
    for scenario in scenarios(42):
        assert values_for(scenario) == values_for(scenario)
        sample = sample_factory(scenario)
        validate_sample(sample)
        for value in sample.labels.pii_values:
            assert value_key(value.canonical_value, value.slot.entity_type) == value_key(
                value.spoken_form, value.slot.entity_type
            )
            if scenario.channel == "voice" and value.slot.entity_type != "PERSON":
                assert not any(char.isdigit() for char in value.surface_form)
        assert sample.conversation == insert_pii(draft_factory(scenario), scenario)[0]


def test_offsets_cover_unicode_and_repeated_values(draft_factory: Callable) -> None:
    scenario = scenarios(42)[1]
    draft = draft_factory(scenario)
    token = scenario.slots[0].token
    turn = DraftTurn(speaker="customer", text=f"Café — {token}; again {token}.")
    updated = draft.model_copy(update={"turns": [turn, *draft.turns[1:]]})
    conversation, values, spans = insert_pii(updated, scenario)
    repeated = [span for span in spans if span.slot_token == token]
    assert len(repeated) == 2
    for span in repeated:
        assert (
            conversation.turns[span.turn_index].text[span.start : span.end]
            == values[0].surface_form
        )
    assert repeated[0].start == len("Café — ")


@pytest.mark.parametrize("field,value", [("start", -1), ("end", 99999), ("turn_index", 100)])
def test_invalid_offsets_rejected(sample_factory: Callable, field: str, value: int) -> None:
    sample = sample_factory(scenarios(42)[1])
    data = sample.model_dump()
    data["labels"]["pii_entities"][0][field] = value
    with pytest.raises(ValidationError):
        Sample.model_validate(data)


def test_asr_value_normalization() -> None:
    assert value_key("four two double four", "CREDIT_CARD") == "4244"
    assert (
        value_key("u s e r one two at e x a m p l e dot c o m", "EMAIL_ADDRESS")
        == "user12@example.com"
    )
    assert value_key("G B eight two W E S T", "IBAN_CODE") == "gb82west"
    assert value_key("+1 (202) 555-0100", "PHONE_NUMBER") == "12025550100"
