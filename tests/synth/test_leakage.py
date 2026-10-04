from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from convintel.contracts import Conversation, Speaker, Turn
from scripts.synth.audit import exact_test_matches
from scripts.synth.scenarios import scenarios
from scripts.synth.storage import read_jsonl


class Utterance(BaseModel):
    text: str


def test_generated_customer_turns_do_not_match_banking77_test(sample_factory: Callable) -> None:
    reference = read_jsonl(Path(__file__).with_name("banking77_test_fixture.jsonl"), Utterance)
    utterances = [row.text for row in reference]
    conversations = [sample_factory(s).conversation for s in scenarios(42)]
    root = Path(__file__).resolve().parents[2]
    for path in (root / "data/synthetic").glob("**/conversations.jsonl"):
        conversations.extend(read_jsonl(path, Conversation))
    assert exact_test_matches(conversations, utterances) == []
    original = conversations[0]
    copied = Turn(index=0, speaker=Speaker.CUSTOMER, text="  HOW DO I LOCATE MY CARD?  ")
    contaminated = original.model_copy(update={"turns": (copied, *original.turns[1:])})
    matches = exact_test_matches([contaminated], utterances)
    assert len(matches) == 1
    assert matches[0].turn_index == 0


def test_generator_has_no_holdout_or_example_dependency() -> None:
    root = Path(__file__).resolve().parents[2]
    generation = (root / "scripts/synth/generate.py").read_text()
    prompt = "\n".join(
        path.read_text()
        for path in (
            *sorted((root / "prompts/synthetic").glob("v*.yaml")),
            root / "prompts/synthetic_judge/v1.yaml",
            root / "scripts/synth/scenarios.py",
        )
    )
    reference = read_jsonl(Path(__file__).with_name("banking77_test_fixture.jsonl"), Utterance)
    assert "from .audit" not in generation
    assert "banking77_test_fixture" not in generation
    assert all(row.text not in generation + prompt for row in reference)
