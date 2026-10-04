"""Temperature scaling and calibration error, in plain numpy.

Shared by training (fit T on validation, report ECE) and the operator (apply T at inference),
so the exact same softmax runs in both places and CI tests it without torch.

Why temperature scaling (Guo et al., 2017): fine-tuned transformers are over-confident, and
ADR-0001 routes on the confidence value itself. A single scalar T keeps the argmax (accuracy
and macro-F1 are unchanged) and is fitted on held-out data with one parameter, so it cannot
overfit the validation split the way per-class methods can.
"""

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

FloatArray = NDArray[np.float64]

DEFAULT_N_BINS = 15
"""Equal-width bins, as in Guo et al.; comparable with published ECE numbers."""

TEMPERATURE_BOUNDS = (0.05, 20.0)


def softmax(logits: NDArray[np.floating], temperature: float = 1.0) -> FloatArray:
    """Row-wise softmax of `logits / temperature`, numerically stable."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


def negative_log_likelihood(
    logits: NDArray[np.floating], labels: NDArray[np.integer], temperature: float
) -> float:
    """Mean NLL of the true labels under `softmax(logits / temperature)`."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def fit_temperature(
    logits: NDArray[np.floating],
    labels: NDArray[np.integer],
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
    tol: float = 1e-5,
) -> float:
    """Temperature minimizing validation NLL, by golden-section search on log T.

    One parameter, so a bracketed 1-D search is enough and needs no optimizer dependency.
    Searching log T keeps the steps proportional on both sides of T = 1.
    """
    if len(labels) == 0:
        raise ValueError("cannot fit a temperature on an empty set")
    inv_phi = (math.sqrt(5) - 1) / 2
    lo, hi = math.log(bounds[0]), math.log(bounds[1])

    def loss(log_t: float) -> float:
        return negative_log_likelihood(logits, labels, math.exp(log_t))

    c, d = hi - inv_phi * (hi - lo), lo + inv_phi * (hi - lo)
    fc, fd = loss(c), loss(d)
    while hi - lo > tol:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - inv_phi * (hi - lo)
            fc = loss(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + inv_phi * (hi - lo)
            fd = loss(d)
    return math.exp((lo + hi) / 2)


class ReliabilityBin(BaseModel):
    """One bar of a reliability diagram: mean confidence vs accuracy of its predictions."""

    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)
    count: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1, description="Mean top-1 probability; 0 if empty")
    accuracy: float = Field(ge=0, le=1, description="Top-1 accuracy; 0 if empty")


def reliability_bins(
    probs: NDArray[np.floating], labels: NDArray[np.integer], n_bins: int = DEFAULT_N_BINS
) -> list[ReliabilityBin]:
    """Top-1 predictions grouped into equal-width confidence bins (right-closed)."""
    confidences = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin i covers (edges[i], edges[i+1]]; confidence is always > 0 after a softmax.
    index = np.clip(np.searchsorted(edges, confidences, side="left") - 1, 0, n_bins - 1)
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        mask = index == i
        count = int(mask.sum())
        bins.append(
            ReliabilityBin(
                lower=float(edges[i]),
                upper=float(edges[i + 1]),
                count=count,
                confidence=float(confidences[mask].mean()) if count else 0.0,
                accuracy=float(correct[mask].mean()) if count else 0.0,
            )
        )
    return bins


def expected_calibration_error(
    probs: NDArray[np.floating], labels: NDArray[np.integer], n_bins: int = DEFAULT_N_BINS
) -> float:
    """ECE: count-weighted mean |accuracy - confidence| over the reliability bins."""
    total = len(labels)
    if total == 0:
        raise ValueError("cannot compute ECE on an empty set")
    bins = reliability_bins(probs, labels, n_bins)
    return sum(b.count / total * abs(b.accuracy - b.confidence) for b in bins)
