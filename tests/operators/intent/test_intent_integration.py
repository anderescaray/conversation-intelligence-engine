"""The registered champion, loaded from the MLflow server of `infra/docker-compose.yml`.

Needs `docker compose up`, a trained and promoted model (training/intent/README.md) and the
`ml` group. The conversation is synthetic.
"""

from pathlib import Path

import pytest
from intent_fakes import make_conversation

from convintel.contracts import Speaker
from convintel.operators.intent import DEFAULT_MODEL_URI, IntentOperator

pytestmark = pytest.mark.integration

C, A = Speaker.CUSTOMER, Speaker.AGENT


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
async def test_champion_from_mlflow_classifies_a_lost_card(device: str, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("mlflow")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    operator = IntentOperator(model_uri=DEFAULT_MODEL_URI, cache_dir=tmp_path, device=device)

    await operator.warmup()
    result = await operator.run(
        make_conversation(
            (C, "Hi, good morning."),
            (A, "Good morning, how can I help you today?"),
            (C, "I think someone stole my card, I can't find it anywhere."),
        )
    )

    assert result.output.label == "lost_or_stolen_card"
    assert result.output.source == "classifier"
    assert result.output.top_k[0][0] == "lost_or_stolen_card"
    assert result.model_id is not None and result.model_id.startswith("intent-classifier/v")
    assert operator.threshold is not None
