"""Classification metrics and the coverage/accuracy threshold study (numpy only).

The threshold τ is the ADR-0001 routing point: predictions with calibrated confidence ≥ τ are
served by the classifier, the rest go to the LLM verifier. It is chosen on validation and only
reported on test.
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

TARGET_COVERED_ACCURACY = 0.97
MIN_COVERED_CASES = 50
"""Below this many served cases the covered accuracy is too noisy to pick τ from: with 50
cases and 97 % true accuracy, a single error moves it by 2 points."""
THRESHOLD_GRID_STEP = 0.001


def accuracy(predictions: NDArray[np.integer], labels: NDArray[np.integer]) -> float:
    if len(labels) == 0:
        raise ValueError("cannot score an empty set")
    return float((predictions == labels).mean())


def macro_f1(
    predictions: NDArray[np.integer], labels: NDArray[np.integer], num_classes: int
) -> float:
    """Unweighted mean F1 over all `num_classes` classes; a class never predicted nor seen
    scores 0, so a missing class drags the mean down instead of being silently dropped."""
    if len(labels) == 0:
        raise ValueError("cannot score an empty set")
    f1s: list[float] = []
    for c in range(num_classes):
        tp = int(((predictions == c) & (labels == c)).sum())
        fp = int(((predictions == c) & (labels != c)).sum())
        fn = int(((predictions != c) & (labels == c)).sum())
        denominator = 2 * tp + fp + fn
        f1s.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(f1s))


class CoveragePoint(BaseModel):
    """Serving only predictions with confidence ≥ `threshold`."""

    threshold: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1, description="Fraction of cases served")
    served: int = Field(ge=0, description="Number of cases served")
    accuracy: float | None = Field(ge=0, le=1, description="Accuracy on served cases")


def coverage_accuracy_curve(
    confidences: NDArray[np.floating],
    correct: NDArray[np.bool_],
    step: float = THRESHOLD_GRID_STEP,
) -> list[CoveragePoint]:
    """Coverage and covered accuracy for thresholds 0, step, 2*step, ..., 1."""
    n_steps = round(1 / step)
    return [point_at(confidences, correct, i / n_steps) for i in range(n_steps + 1)]


def choose_threshold(
    curve: list[CoveragePoint],
    target_accuracy: float = TARGET_COVERED_ACCURACY,
    min_served: int = MIN_COVERED_CASES,
) -> CoveragePoint | None:
    """Highest-coverage point serving at least `min_served` cases whose covered accuracy meets
    the target; None if none does.

    Covered accuracy is not monotone in τ on a finite sample, so this scans every point rather
    than stopping at the first one that crosses the target. Ties in coverage (thresholds in the
    gap between two observed confidences) take the highest threshold: on new data it admits
    the fewest extra cases, so it is the conservative side of the gap.
    """
    eligible = [
        p
        for p in curve
        if p.served >= min_served and p.accuracy is not None and p.accuracy >= target_accuracy
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda p: (p.coverage, p.threshold))


def point_at(
    confidences: NDArray[np.floating], correct: NDArray[np.bool_], threshold: float
) -> CoveragePoint:
    """Coverage and covered accuracy at one fixed threshold (e.g. τ applied to test)."""
    if len(confidences) == 0:
        raise ValueError("cannot evaluate a threshold on an empty set")
    served = confidences >= threshold
    n_served = int(served.sum())
    return CoveragePoint(
        threshold=threshold,
        coverage=n_served / len(confidences),
        served=n_served,
        accuracy=float(correct[served].mean()) if n_served else None,
    )
