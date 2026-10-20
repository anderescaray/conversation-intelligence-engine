from collections.abc import Callable

import pytest

from scripts.synth.scenarios import scenarios
from scripts.synth.schema import DraftTurn
from scripts.synth.validation import Rejected, ensure_unique, validate_draft, validate_sample


@pytest.mark.parametrize(
    "text",
    [
        "Write to invented@example.com.",
        "Call 202-555-0133.",
        "My name is John Smith.",
        "My card is four two four two four two.",
        "My account is 99887766.",
    ],
)
def test_model_cannot_invent_pii(draft_factory: Callable, text: str) -> None:
    scenario = scenarios(42)[0]
    draft = draft_factory(scenario)
    draft.turns[0] = DraftTurn(speaker="customer", text=text)
    with pytest.raises(Rejected, match="unowned_pii"):
        validate_draft(draft, scenario)


@pytest.mark.parametrize("text", ["Hello [[UNKNOWN_1]]", "Hello {{name}}", "Hello <PERSON>"])
def test_unknown_or_malformed_placeholders_rejected(draft_factory: Callable, text: str) -> None:
    scenario = scenarios(42)[0]
    draft = draft_factory(scenario)
    draft.turns[0] = DraftTurn(speaker="customer", text=text)
    with pytest.raises(Rejected):
        validate_draft(draft, scenario)


def test_missing_placeholder_and_escalation_rejected(draft_factory: Callable) -> None:
    scenario = scenarios(42)[13]
    assert scenario.escalation_requested and scenario.slots
    draft = draft_factory(scenario)
    draft.turns[0] = DraftTurn(speaker="customer", text="My issue is still open.")
    with pytest.raises(Rejected, match="placeholder_inventory"):
        validate_draft(draft, scenario)
    draft = draft_factory(scenario)
    draft.turns[0] = draft.turns[0].model_copy(
        update={
            "text": draft.turns[0].text.replace(
                " I am frustrated and want to speak to a supervisor.", ""
            )
        }
    )
    with pytest.raises(Rejected, match="escalation_evidence"):
        validate_draft(draft, scenario)


def test_ner_is_injectable_and_only_sees_masked_text(sample_factory: Callable) -> None:
    sample = sample_factory(scenarios(42)[1])
    received: list[str] = []

    def scanner(text: str) -> bool:
        received.append(text)
        return False

    validate_sample(sample, scanner)
    assert all(v.canonical_value not in " ".join(received) for v in sample.labels.pii_values)
    with pytest.raises(Rejected, match="unowned_pii"):
        validate_sample(sample, lambda _: True)


def test_pii_changes_do_not_evade_deduplication(sample_factory: Callable) -> None:
    first = sample_factory(scenarios(42)[1])
    second = sample_factory(first.labels.scenario.model_copy(update={"sample_seed": 123}))
    with pytest.raises(Rejected, match="duplicate_dialogue"):
        ensure_unique(second, [first])
