import pytest
from sentiment_fakes import NEG, NEU, POS

from convintel.contracts import Sentiment
from convintel.operators.sentiment import (
    SentimentScores,
    overall_sentiment,
    recency_weights,
    weighted_polarity,
)


def test_last_turn_weighs_one_and_weights_halve_every_half_life() -> None:
    assert recency_weights(4, half_life=2) == pytest.approx([0.5**1.5, 0.5, 0.5**0.5, 1.0])
    assert recency_weights(4)[-1] == 1.0


def test_weights_do_not_depend_on_length_for_the_last_turns() -> None:
    assert recency_weights(3)[-3:] == pytest.approx(recency_weights(10)[-3:])


def test_no_turns_is_neutral() -> None:
    assert weighted_polarity([]) == 0.0
    assert overall_sentiment([]) is Sentiment.NEUTRAL


@pytest.mark.parametrize(
    ("trajectory", "expected"),
    [
        ([NEG], Sentiment.NEGATIVE),
        ([POS], Sentiment.POSITIVE),
        ([NEU, NEU, NEU], Sentiment.NEUTRAL),
        # Angry opener, resolved at the end: the outcome wins.
        ([NEG, NEG, NEU, POS, POS], Sentiment.POSITIVE),
        # Polite opener, escalating at the end.
        ([POS, POS, NEU, NEG, NEG], Sentiment.NEGATIVE),
        # Closing formality does not erase the frustration just before it.
        ([NEG, NEG, NEG, NEU], Sentiment.NEGATIVE),
    ],
)
def test_overall_follows_the_end_of_the_trajectory(
    trajectory: list[SentimentScores], expected: Sentiment
) -> None:
    assert overall_sentiment(trajectory) is expected


def test_last_turn_alone_does_not_flip_a_long_consistent_history() -> None:
    assert weighted_polarity([NEG] * 20 + [POS]) < 0


def test_last_two_agreeing_turns_outweigh_any_history() -> None:
    assert weighted_polarity([NEG] * 20 + [POS, POS]) > 0


def test_plain_average_would_disagree_on_recovery() -> None:
    """The case that motivates recency weighting: a plain mean calls this neutral."""
    trajectory = [NEG, NEG, NEG, POS, POS]
    plain_mean = sum(s.polarity for s in trajectory) / len(trajectory)

    assert abs(plain_mean) < 0.2
    assert overall_sentiment(trajectory) is Sentiment.POSITIVE


def test_hesitant_turns_count_less_than_confident_ones() -> None:
    hesitant_negative = SentimentScores(negative=0.4, neutral=0.35, positive=0.25)

    assert hesitant_negative.label is Sentiment.NEGATIVE
    assert overall_sentiment([hesitant_negative]) is Sentiment.NEUTRAL


def test_label_is_argmax() -> None:
    assert NEG.label is Sentiment.NEGATIVE
    assert NEU.label is Sentiment.NEUTRAL
    assert POS.label is Sentiment.POSITIVE
