import json
from collections.abc import Callable
from pathlib import Path

import pytest

from convintel.contracts import Speaker
from convintel.errors import ConvIntelError
from scripts.synth.predict_flips import predict_source, predicted_fields
from scripts.synth.scenarios import scenarios
from scripts.synth.schema import Sample, Scenario
from scripts.synth.storage import write_jsonl
from scripts.synth.taxonomy import INTENT_DESCRIPTIONS_V1, INTENT_DESCRIPTIONS_V2

C, A = Speaker.CUSTOMER, Speaker.AGENT
UNCHANGED = next(
    i
    for i in INTENT_DESCRIPTIONS_V1
    if i in INTENT_DESCRIPTIONS_V2 and (INTENT_DESCRIPTIONS_V1[i] == INTENT_DESCRIPTIONS_V2[i])
)
CHANGED = "get_physical_card"


def fields(
    turns: list[tuple[Speaker, str]],
    *,
    intent: str = UNCHANGED,
    resolved: bool = True,
    start: str = "neutral",
    end: str = "neutral",
) -> tuple[str, ...]:
    return predicted_fields(
        intent=intent, resolved=resolved, start_sentiment=start, end_sentiment=end, turns=turns
    )


CALM = [
    (C, "My transfer from Monday has not arrived yet. Could you check it?"),
    (A, "I can see it; it was held for a routine check."),
    (C, "Ok, thanks."),
    (A, "You're welcome."),
]


def test_nothing_flips_when_targets_match_the_new_definitions() -> None:
    assert fields(CALM) == ()


def test_intent_flip_is_an_upper_bound_on_changed_descriptions() -> None:
    assert fields(CALM, intent=CHANGED) == ("intent",)


def test_calm_opening_with_a_negative_target_flips_and_mild_emotion_does_not() -> None:
    assert fields(CALM, start="negative") == ("start_sentiment",)
    mild = [(C, "My transfer is late, which is a little irritating."), *CALM[1:]]
    assert fields(mild, start="negative") == ()
    assert fields(mild, start="neutral") == ("start_sentiment",)


def test_exclamation_marks_alone_are_not_negative_emotion() -> None:
    greeting = [(C, "Hi! My transfer from Monday has not arrived yet."), *CALM[1:]]
    assert fields(greeting, start="neutral") == ()


def test_courtesy_closing_with_a_positive_target_flips() -> None:
    assert fields(CALM, end="positive") == ("end_sentiment",)
    relieved = [*CALM[:2], (C, "That's a relief, I really appreciate it."), CALM[3]]
    assert fields(relieved, end="positive") == ()


def test_completed_action_with_an_unresolved_target_flips_and_deferral_does_not() -> None:
    done = [*CALM[:3], (A, "I've processed your refund and it is on its way.")]
    assert fields(done, resolved=False) == ("resolved",)
    deferred = [*CALM[:3], (A, "I've passed it on; the team will review it and contact you.")]
    assert fields(deferred, resolved=False) == ()
    assert fields(deferred, resolved=True) == ("resolved",)


def test_a_candidate_without_both_speakers_is_rejected() -> None:
    with pytest.raises(ConvIntelError, match="customer and agent"):
        fields([(C, "Hello?")])


def test_prediction_report_counts_fields_and_never_writes_text(
    tmp_path: Path, sample_factory: Callable[[Scenario], Sample]
) -> None:
    samples = [sample_factory(s) for s in scenarios(42)[:3]]
    source = tmp_path / "candidates"
    source.mkdir()
    write_jsonl(source / "candidates.jsonl", samples)
    output = tmp_path / "prediction.json"
    report = predict_source(source, output)
    assert report.total == 3 and set(report.predicted_by_field) >= {"intent", "resolved"}
    assert report.intent_description_changed == report.predicted_by_field["intent"]
    recorded = output.read_text()
    for sample in samples:
        assert all(turn.text not in recorded for turn in sample.conversation.turns)
    assert json.loads(recorded)["predictions"][0]["id"] == samples[0].conversation.id
    with pytest.raises(ConvIntelError, match="Output exists"):
        predict_source(source, output)
