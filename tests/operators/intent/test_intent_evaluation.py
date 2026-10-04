import numpy as np
import pytest

from convintel.operators.intent.evaluation import (
    CoveragePoint,
    accuracy,
    choose_threshold,
    coverage_accuracy_curve,
    macro_f1,
    point_at,
)


def test_macro_f1_matches_a_hand_computation() -> None:
    labels = np.array([0, 0, 1, 1, 2])
    predictions = np.array([0, 1, 1, 1, 0])
    # class 0: tp=1 fp=1 fn=1 -> 0.5; class 1: tp=2 fp=1 fn=0 -> 0.8; class 2: 0
    assert macro_f1(predictions, labels, 3) == pytest.approx((0.5 + 0.8 + 0.0) / 3)
    assert accuracy(predictions, labels) == pytest.approx(0.6)


def test_a_class_absent_everywhere_still_counts_as_zero() -> None:
    labels = predictions = np.array([0, 1])

    assert macro_f1(predictions, labels, 2) == 1.0
    assert macro_f1(predictions, labels, 4) == 0.5


def test_metrics_reject_empty_sets() -> None:
    empty = np.array([], dtype=np.int64)
    with pytest.raises(ValueError):
        macro_f1(empty, empty, 3)
    with pytest.raises(ValueError):
        accuracy(empty, empty)


def test_curve_starts_at_full_coverage_and_never_grows() -> None:
    confidences = np.array([0.2, 0.5, 0.8, 0.95])
    correct = np.array([False, True, True, True])

    curve = coverage_accuracy_curve(confidences, correct, step=0.05)

    assert curve[0] == CoveragePoint(threshold=0.0, coverage=1.0, served=4, accuracy=0.75)
    assert curve[-1].threshold == 1.0
    assert curve[-1].coverage == 0.0 and curve[-1].accuracy is None
    coverages = [p.coverage for p in curve]
    assert coverages == sorted(coverages, reverse=True)


def test_choose_threshold_maximizes_coverage_at_the_target() -> None:
    confidences = np.array([0.3, 0.6, 0.7, 0.9, 0.95])
    correct = np.array([False, True, False, True, True])
    curve = coverage_accuracy_curve(confidences, correct, step=0.05)

    chosen = choose_threshold(curve, target_accuracy=0.97, min_served=1)

    assert chosen is not None
    assert chosen.coverage == pytest.approx(0.4)
    assert chosen.accuracy == 1.0
    # Ties in coverage take the highest threshold still serving the same cases.
    assert chosen.threshold == pytest.approx(0.9)


def test_choose_threshold_scans_past_non_monotone_accuracy() -> None:
    curve = [
        CoveragePoint(threshold=0.5, coverage=0.9, served=90, accuracy=0.98),
        CoveragePoint(threshold=0.6, coverage=0.8, served=80, accuracy=0.96),
        CoveragePoint(threshold=0.7, coverage=0.7, served=70, accuracy=0.99),
    ]

    assert choose_threshold(curve, 0.97) == curve[0]


def test_choose_threshold_ignores_points_with_too_few_cases() -> None:
    curve = [
        CoveragePoint(threshold=0.5, coverage=0.5, served=50, accuracy=0.9),
        CoveragePoint(threshold=0.9, coverage=0.02, served=2, accuracy=1.0),
    ]

    assert choose_threshold(curve, 0.97, min_served=50) is None
    assert choose_threshold(curve, 0.97, min_served=2) == curve[1]


def test_choose_threshold_is_none_when_the_target_is_unreachable() -> None:
    curve = coverage_accuracy_curve(np.array([0.9, 0.95]), np.array([False, False]))

    assert choose_threshold(curve, 0.97) is None


def test_point_at_a_fixed_threshold() -> None:
    point = point_at(np.array([0.4, 0.8, 0.9]), np.array([True, False, True]), 0.8)

    assert (point.threshold, point.coverage, point.served, point.accuracy) == (
        0.8,
        pytest.approx(2 / 3),
        2,
        0.5,
    )
