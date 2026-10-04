"""Overall sentiment from the customer's per-turn trajectory.

Why recency weighting: the question downstream consumers ask is "how did the customer leave?"
(churn risk, escalation, CSAT proxy). A call that opens angry and ends with "great, thanks,
that fixed it" is a success; a plain average would call it neutral. The opposite case, polite
at first and furious at the end, is the one most worth flagging.

Why exponential decay with a half-life in turns, rather than "last turn only" or linear
weights: the last turn alone is often a closing formality ("ok bye"), so the turns just before
it must still count; linear weights let early turns of long calls dominate by sheer number.
Exponential weights do not depend on conversation length.

The half-life h follows from two requirements. With r = 2**(-1/h) the weight ratio between
consecutive turns, and an arbitrarily long history before the final turns:
- the final turn alone must not outweigh a consistent history: 1 < r / (1 - r), so h > 1;
- the final two turns, when they agree, must outweigh any history: 1 + r > r**2 / (1 - r),
  so r < 1/sqrt(2), i.e. h < 2.
h = 1.5 is the middle of that range. ("Outweigh" is about the sign of the weighted polarity;
whether it clears the neutral band also depends on how confident the turns are.)

Turns contribute their polarity (P(positive) - P(negative)) rather than their argmax label, so
a hesitant "negative" at 0.4 counts less than a clear one at 0.95.
"""

from collections.abc import Sequence

from convintel.contracts import Sentiment

from .model import SentimentScores

HALF_LIFE_TURNS = 1.5
NEUTRAL_BAND = 0.2
"""|weighted polarity| below this is neutral. Initial value; tune on the T06 golden set."""


def recency_weights(n: int, half_life: float = HALF_LIFE_TURNS) -> list[float]:
    """Weights for turns 0..n-1; the last turn weighs 1 and weight halves every `half_life`."""
    return [0.5 ** ((n - 1 - i) / half_life) for i in range(n)]


def weighted_polarity(scores: Sequence[SentimentScores]) -> float:
    """Recency-weighted mean polarity in [-1, 1]; 0 when there are no turns."""
    if not scores:
        return 0.0
    weights = recency_weights(len(scores))
    return sum(w * s.polarity for w, s in zip(weights, scores, strict=True)) / sum(weights)


def overall_sentiment(scores: Sequence[SentimentScores]) -> Sentiment:
    polarity = weighted_polarity(scores)
    if polarity >= NEUTRAL_BAND:
        return Sentiment.POSITIVE
    if polarity <= -NEUTRAL_BAND:
        return Sentiment.NEGATIVE
    return Sentiment.NEUTRAL
