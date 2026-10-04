"""Calibration study (validation) and frozen evaluation (test), from logits only.

Split in two on purpose: `calibration_study` is the only place that chooses anything (T, τ)
and is fed validation logits; `evaluate_frozen` takes T and τ as inputs and only measures.
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from convintel.operators.intent.calibration import (
    ReliabilityBin,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    reliability_bins,
    softmax,
)
from convintel.operators.intent.evaluation import (
    CoveragePoint,
    accuracy,
    choose_threshold,
    coverage_accuracy_curve,
    macro_f1,
    point_at,
)


class CalibrationStudy(BaseModel):
    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    bins_before: list[ReliabilityBin]
    bins_after: list[ReliabilityBin]
    curve: list[CoveragePoint]
    threshold: CoveragePoint | None
    """None when no threshold reaches the target covered accuracy."""


class FrozenEvaluation(BaseModel):
    n: int
    accuracy: float
    macro_f1: float
    ece_before: float
    ece_after: float
    bins_before: list[ReliabilityBin]
    bins_after: list[ReliabilityBin]
    curve: list[CoveragePoint]
    at_threshold: CoveragePoint | None


def calibration_study(
    logits: NDArray[np.floating],
    labels: NDArray[np.int64],
    n_bins: int,
    target_accuracy: float,
    min_covered: int,
) -> CalibrationStudy:
    """Fit T and choose τ. Call with validation data only."""
    temperature = fit_temperature(logits, labels)
    raw, calibrated = softmax(logits), softmax(logits, temperature)
    curve = coverage_accuracy_curve(calibrated.max(axis=1), calibrated.argmax(axis=1) == labels)
    return CalibrationStudy(
        temperature=temperature,
        nll_before=negative_log_likelihood(logits, labels, 1.0),
        nll_after=negative_log_likelihood(logits, labels, temperature),
        ece_before=expected_calibration_error(raw, labels, n_bins),
        ece_after=expected_calibration_error(calibrated, labels, n_bins),
        bins_before=reliability_bins(raw, labels, n_bins),
        bins_after=reliability_bins(calibrated, labels, n_bins),
        curve=curve,
        threshold=choose_threshold(curve, target_accuracy, min_covered),
    )


def evaluate_frozen(
    logits: NDArray[np.floating],
    labels: NDArray[np.int64],
    temperature: float,
    threshold: float | None,
    n_bins: int,
    num_classes: int,
) -> FrozenEvaluation:
    """Measure a frozen model + T + τ. Chooses nothing, so it is safe to run on test."""
    raw, calibrated = softmax(logits), softmax(logits, temperature)
    predictions = calibrated.argmax(axis=1)
    confidences, correct = calibrated.max(axis=1), predictions == labels
    return FrozenEvaluation(
        n=len(labels),
        accuracy=accuracy(predictions, labels),
        macro_f1=macro_f1(predictions, labels, num_classes),
        ece_before=expected_calibration_error(raw, labels, n_bins),
        ece_after=expected_calibration_error(calibrated, labels, n_bins),
        bins_before=reliability_bins(raw, labels, n_bins),
        bins_after=reliability_bins(calibrated, labels, n_bins),
        curve=coverage_accuracy_curve(confidences, correct),
        at_threshold=None if threshold is None else point_at(confidences, correct, threshold),
    )
