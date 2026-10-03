"""Real model from the Hugging Face hub: needs network and torch (`uv sync --group ml`)."""

import pytest
from sentiment_fakes import make_conversation

from convintel.contracts import Sentiment, Speaker
from convintel.operators.sentiment import SentimentOperator

pytestmark = pytest.mark.integration

C, A = Speaker.CUSTOMER, Speaker.AGENT


async def test_real_model_tracks_a_recovering_customer() -> None:
    pytest.importorskip("torch")
    conversation = make_conversation(
        (C, "This is the third time my internet has dropped today and I am furious."),
        (A, "I'm sorry about that. Let me check the line."),
        (C, "Fine. The router lights are blinking orange."),
        (A, "I've pushed a fix to your router, can you check now?"),
        (C, "It works now, thank you so much, that's great!"),
    )

    result = await SentimentOperator().run(conversation)

    trajectory = result.output.customer_trajectory
    assert len(trajectory) == 3
    assert trajectory[0] is Sentiment.NEGATIVE
    assert trajectory[-1] is Sentiment.POSITIVE
    assert result.output.overall is Sentiment.POSITIVE
    assert result.model_id == "cardiffnlp/twitter-roberta-base-sentiment-latest"
