"""The real transformers adapter on a tiny, randomly initialized DistilBERT built in the test.

Needs the `ml` group (skipped in CI) but no network: config and vocabulary are local.
"""

from pathlib import Path

import numpy as np
import pytest
from intent_fakes import make_conversation

from convintel.contracts import Speaker
from convintel.operators.intent import (
    IntentBundleMeta,
    IntentModelUnavailableError,
    IntentOperator,
    load_bundle,
)
from convintel.operators.intent.bundle import write_bundle_meta

LABELS = ("card_arrival", "lost_or_stolen_card", "exchange_rate")
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "my", "card", "is", "lost", "rate"]


def _tiny_bundle(directory: Path, labels: tuple[str, ...] = LABELS) -> Path:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    directory.mkdir(parents=True, exist_ok=True)
    vocab = directory / "vocab.txt"
    vocab.write_text("\n".join(VOCAB))
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    config = transformers.DistilBertConfig(
        vocab_size=len(VOCAB),
        dim=16,
        hidden_dim=32,
        n_layers=1,
        n_heads=2,
        max_position_embeddings=32,
        num_labels=3,
        id2label=dict(enumerate(LABELS)),
        label2id={name: i for i, name in enumerate(LABELS)},
    )
    model = transformers.DistilBertForSequenceClassification(config)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    write_bundle_meta(
        directory,
        IntentBundleMeta(
            labels=labels,
            temperature=1.5,
            threshold=0.5,
            max_length=8,
            base_model="tiny-distilbert",
            dataset_version="synthetic@0",
        ),
    )
    return directory


def test_load_bundle_and_predict_batches(tmp_path: Path) -> None:
    bundle = _tiny_bundle(tmp_path / "bundle")

    model = load_bundle(bundle, model_id="tiny", device="cpu")
    texts = ["my card is lost", "rate", "my card", "lost lost lost lost lost lost lost lost lost"]
    logits = model.predict_logits(texts)

    assert logits.shape == (4, 3)
    assert logits.dtype == np.float32
    assert model.labels == LABELS
    assert (model.temperature, model.threshold) == (1.5, 0.5)
    assert model.predict_logits([]).shape == (0, 3)


def test_label_count_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle = _tiny_bundle(tmp_path / "bundle", labels=("a", "b"))

    with pytest.raises(IntentModelUnavailableError, match="3 logits but 2 labels"):
        load_bundle(bundle, model_id="tiny", device="cpu")


async def test_operator_end_to_end_on_a_local_bundle(tmp_path: Path) -> None:
    bundle = _tiny_bundle(tmp_path / "bundle")
    operator = IntentOperator(model_uri=str(bundle), device="cpu")

    await operator.warmup()
    result = await operator.run(make_conversation((Speaker.CUSTOMER, "my card is lost")))

    assert result.output.label in LABELS
    assert len(result.output.top_k) == 3
    assert result.model_id == "local/bundle"
    assert operator.threshold == 0.5
