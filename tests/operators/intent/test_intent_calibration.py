import numpy as np
import pytest

from convintel.operators.intent.calibration import (
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    reliability_bins,
    softmax,
)


def _sample(
    n: int, n_classes: int, true_temperature: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Logits whose labels are drawn from softmax(logits / true_temperature)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 4, size=(n, n_classes))
    probs = softmax(logits, true_temperature)
    labels = np.array([rng.choice(n_classes, p=p) for p in probs], dtype=np.int64)
    return logits, labels


def test_softmax_rows_sum_to_one_and_ignore_shifts() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])

    probs = softmax(logits)

    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_allclose(probs[0], probs[1])


def test_higher_temperature_flattens_without_changing_the_argmax() -> None:
    logits = np.array([[0.5, 3.0, 1.0]])

    sharp, flat = softmax(logits, 0.5), softmax(logits, 4.0)

    assert sharp.max() > softmax(logits).max() > flat.max()
    assert sharp.argmax() == flat.argmax() == 1


def test_softmax_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="positive"):
        softmax(np.zeros((1, 2)), 0.0)


@pytest.mark.parametrize("true_temperature", [0.5, 1.0, 2.5])
def test_fit_temperature_recovers_the_generating_temperature(true_temperature: float) -> None:
    logits, labels = _sample(8000, 5, true_temperature)

    fitted = fit_temperature(logits, labels)

    assert fitted == pytest.approx(true_temperature, rel=0.1)


def test_fitted_temperature_minimizes_nll() -> None:
    logits, labels = _sample(2000, 4, 2.0)

    fitted = fit_temperature(logits, labels)

    best = negative_log_likelihood(logits, labels, fitted)
    assert best <= negative_log_likelihood(logits, labels, fitted * 1.2)
    assert best <= negative_log_likelihood(logits, labels, fitted / 1.2)


def test_fit_temperature_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_temperature(np.zeros((0, 3)), np.zeros(0, dtype=np.int64))


def test_ece_is_zero_when_confident_and_right() -> None:
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert expected_calibration_error(probs, np.array([0, 1])) == pytest.approx(0.0)


def test_ece_is_the_confidence_accuracy_gap() -> None:
    probs = np.array([[0.9, 0.1]] * 10)
    labels = np.array([0] * 5 + [1] * 5)

    assert expected_calibration_error(probs, labels) == pytest.approx(0.4)


def test_temperature_scaling_reduces_ece_of_an_overconfident_model() -> None:
    logits, labels = _sample(5000, 5, 3.0)

    before = expected_calibration_error(softmax(logits), labels)
    after = expected_calibration_error(softmax(logits, fit_temperature(logits, labels)), labels)

    assert after < before / 3


def test_reliability_bins_cover_every_prediction_and_are_right_closed() -> None:
    probs = np.array([[0.6, 0.4], [0.61, 0.39], [1.0, 0.0]])

    bins = reliability_bins(probs, np.array([0, 1, 0]), n_bins=5)

    assert [b.count for b in bins] == [0, 0, 1, 1, 1]
    assert (bins[2].lower, bins[2].upper) == pytest.approx((0.4, 0.6))
    assert bins[3].accuracy == 0.0
    assert bins[4].confidence == pytest.approx(1.0)
