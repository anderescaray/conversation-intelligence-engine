"""Pure parts of `training/intent/verifier_candidates.py` (no torch, data or network)."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from convintel.operators.intent.operator import pick_intent
from training.intent.verifier_candidates import (
    DEFAULT_STUDY_CONFIG,
    TokenAssumptions,
    VerifierStudyConfig,
    below_threshold,
    candidate_tokens,
    estimate_cost,
    load_study_config,
    rank_candidates,
    recall_at_k,
    string_dicts,
    study_split,
)

LABELS = ("a", "b", "c", "d")
CANDIDATES = [("a", "b", "c", "d"), ("b", "a", "c", "d"), ("c", "d", "a", "b")]
TRUTH = ["a", "a", "a"]  # true label at rank 1, 2 and 3


@pytest.mark.parametrize(("k", "expected"), [(1, 1 / 3), (2, 2 / 3), (3, 1.0), (4, 1.0)])
def test_recall_at_k_counts_the_true_label_within_the_first_k(k: int, expected: float) -> None:
    assert recall_at_k(CANDIDATES, TRUTH, k) == pytest.approx(expected)


def test_recall_at_k_includes_rank_k_and_excludes_rank_k_plus_one() -> None:
    assert recall_at_k([("x", "y", "t", "z")], ["t"], 3) == 1.0
    assert recall_at_k([("x", "y", "z", "t")], ["t"], 3) == 0.0


def test_recall_at_k_of_no_cases_is_undefined() -> None:
    assert recall_at_k([], [], 3) is None


@pytest.mark.parametrize(
    ("candidates", "truth", "k", "match"),
    [
        (CANDIDATES, TRUTH, 0, "at least 1"),
        (CANDIDATES, TRUTH[:2], 1, "length"),
        (CANDIDATES, TRUTH, 5, "fewer than 5"),
    ],
)
def test_recall_at_k_rejects_bad_input(
    candidates: list[tuple[str, ...]], truth: list[str], k: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        recall_at_k(candidates, truth, k)


def test_confidence_equal_to_tau_is_served_not_verified() -> None:
    mask = below_threshold(np.array([0.5, 0.861, 0.8609, 0.9]), 0.861)
    assert mask.tolist() == [True, False, True, False]


def test_rank_candidates_reproduces_the_operator() -> None:
    probs = np.array([[0.1, 0.6, 0.2, 0.1], [0.25, 0.25, 0.4, 0.1]])

    confidences, candidates = rank_candidates(probs, LABELS, top_k=3)

    for row, confidence, ranked in zip(probs, confidences, candidates, strict=True):
        served = pick_intent(row[None, :], LABELS, 3)
        assert confidence == served.confidence
        assert ranked == tuple(label for label, _ in served.top_k)
    assert candidates[0] == ("b", "c", "d")
    assert candidates[1][0] == "c"


def test_string_dicts_resolves_merges_in_order_and_skips_non_literals() -> None:
    source = (
        'A: dict[str, str] = {"x": "one", "y": ("tw" "o")}\n'
        'B = {**A, "y": "override", "z": "three"}\n'
        "C = {1: A}\n"
        'D = {**UNKNOWN, "x": "x"}\n'
        "E = dict(x='1')\n"
    )

    dicts = string_dicts(source)

    assert dicts == {
        "A": {"x": "one", "y": "two"},
        "B": {"x": "one", "y": "override", "z": "three"},
    }


TOKENS = TokenAssumptions(
    fixed_prompt_tokens=100,
    chars_per_token=4.0,
    label_chars_per_token=2.0,
    per_candidate_overhead_tokens=5,
    conversation_tokens=400,
    output_tokens=50,
)
DESCRIPTIONS = {"a": "x" * 40, "b": "y" * 80, "c": "z" * 20, "d": "w" * 4}


def test_candidate_tokens_adds_label_description_and_overhead() -> None:
    # a: 1/2 + 40/4 + 5 = 15.5 ; b: 1/2 + 80/4 + 5 = 25.5
    assert candidate_tokens(["a", "b"], DESCRIPTIONS, TOKENS) == pytest.approx(41.0)


def test_estimate_cost_prices_both_scenarios_with_the_client_table() -> None:
    row = estimate_cost(
        candidates=[("a", "b", "c"), ("b", "a", "c")],
        texts=["q" * 8, "q" * 16],
        k=2,
        share_below=0.1,
        descriptions=DESCRIPTIONS,
        tokens=TOKENS,
        provider="anthropic",
        model="claude-haiku-4-5",
    )

    # shared = 100 + 41 per call; utterance adds 2 and 4 tokens; conversation adds 400
    assert row.calls == 2
    assert row.mean_input_tokens_utterance == pytest.approx(144.0)
    assert row.mean_input_tokens_conversation == pytest.approx(541.0)
    assert row.usd_utterance == pytest.approx((288 * 1.0 + 100 * 5.0) / 1e6)
    assert row.usd_conversation == pytest.approx((1082 * 1.0 + 100 * 5.0) / 1e6)
    assert row.usd_per_1000_conversations == pytest.approx(row.usd_conversation / 2 * 100)


def test_estimate_cost_refuses_an_unpriced_model() -> None:
    with pytest.raises(ValueError, match="no price"):
        estimate_cost([("a",)], ["q"], 1, 1.0, DESCRIPTIONS, TOKENS, "anthropic", "unpriced")


def _config(ks: tuple[int, ...] = (1, 2, 3)) -> VerifierStudyConfig:
    return VerifierStudyConfig(
        ks=ks,
        taxonomy_commit="0" * 40,
        taxonomy_path="t.py",
        taxonomy_name="T",
        llm_provider="anthropic",
        llm_model="claude-haiku-4-5",
        tokens=TOKENS,
    )


def test_study_split_separates_routing_baseline_and_ceiling() -> None:
    confidences = np.array([0.95, 0.90, 0.50, 0.40, 0.30, 0.20])
    candidates = [
        ("a", "b", "c"),  # served, right
        ("b", "a", "c"),  # served, wrong
        ("a", "b", "c"),  # below τ, right
        ("b", "a", "c"),  # below τ, wrong, truth at rank 2
        ("c", "d", "a"),  # below τ, wrong, truth at rank 3
        ("b", "c", "d"),  # below τ, wrong, truth not in top 3
    ]
    truth = ["a"] * 6

    report = study_split(
        "test", confidences, candidates, truth, ["q"] * 6, 0.861, _config(), DESCRIPTIONS
    )

    assert (report.n, report.n_below, report.n_below_wrong) == (6, 4, 3)
    assert report.share_below == pytest.approx(4 / 6)
    assert report.coverage == pytest.approx(2 / 6)
    assert report.accuracy == pytest.approx(2 / 6)
    assert report.accuracy_below == pytest.approx(1 / 4)
    assert report.accuracy_above == pytest.approx(1 / 2)
    assert [(r.k, r.below_tau, r.below_tau_wrong) for r in report.recall] == [
        (1, pytest.approx(1 / 4), 0.0),
        (2, pytest.approx(2 / 4), pytest.approx(1 / 3)),
        (3, pytest.approx(3 / 4), pytest.approx(2 / 3)),
    ]
    assert [c.calls for c in report.cost] == [4, 4, 4]


def test_default_study_config_is_valid_and_strict(tmp_path: Path) -> None:
    config = load_study_config()
    assert config.ks == (1, 2, 3, 5, 10)
    assert config.taxonomy_name == "INTENT_DESCRIPTIONS_V3"

    raw = yaml.safe_load(DEFAULT_STUDY_CONFIG.read_text())
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(raw | {"kss": [1]}))
    with pytest.raises(ValidationError):
        load_study_config(tmp_path / "c.yaml")
