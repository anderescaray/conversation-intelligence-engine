"""Inference latency of the intent classifier on CPU or GPU (p50 / p95 / p99).

    uv run --group ml python -m training.intent.bench_latency --device cpu [--model-uri URI]

Two measurements, each after warm-up:
- `utterance`: one test utterance through tokenizer + model + calibrated softmax (batch 1);
- `operator`: `IntentOperator.run` on a conversation with three customer turns and two agent
  turns, i.e. what the pipeline pays per conversation (thread hop and lock included).
Test utterances are used as realistic text only; nothing is scored or selected here.
With `--log-to-mlflow`, the numbers are logged to the training run of the resolved version.
"""

import argparse
import asyncio
import importlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from convintel.config import get_settings
from convintel.contracts import Channel, Conversation, Speaker, Turn
from convintel.operators.intent import DEFAULT_MODEL_URI, IntentOperator, softmax
from convintel.operators.intent.model import DEFAULT_CACHE_DIR, load_bundle, resolve_model_uri

from .data import load_split


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
    }


def _conversation(i: int, customer_texts: list[str]) -> Conversation:
    turns: list[Turn] = []
    for text in customer_texts:
        turns.append(Turn(index=len(turns), speaker=Speaker.CUSTOMER, text=text))
        turns.append(Turn(index=len(turns), speaker=Speaker.AGENT, text="Let me check that."))
    return Conversation(
        id=f"bench-{i}",
        channel=Channel.CHAT,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        turns=tuple(turns[:-1]),
        redacted=True,
    )


async def _bench_operator(
    operator: IntentOperator, conversations: list[Conversation], warmup: int
) -> list[float]:
    for conversation in conversations[:warmup]:
        await operator.run(conversation)
    samples: list[float] = []
    for conversation in conversations[warmup:]:
        start = time.perf_counter()
        await operator.run(conversation)
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Intent classifier inference latency (CPU or GPU)."
    )
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("-n", type=int, default=500, help="timed samples per measurement")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--log-to-mlflow", action="store_true")
    args = parser.parse_args(argv)

    torch = importlib.import_module("torch")
    if args.threads:
        torch.set_num_threads(args.threads)
    tracking_uri = get_settings().mlflow_tracking_uri
    directory, model_id = resolve_model_uri(args.model_uri, args.cache_dir, tracking_uri)
    model = load_bundle(directory, model_id, device=args.device)

    texts = load_split("test").texts
    total = args.warmup + args.n
    rng = np.random.default_rng(0)
    picks = [texts[i] for i in rng.choice(len(texts), size=3 * total, replace=True)]

    utterance_ms: list[float] = []
    for i, text in enumerate(picks[:total]):
        start = time.perf_counter()
        softmax(model.predict_logits([text]), model.temperature)
        if i >= args.warmup:
            utterance_ms.append((time.perf_counter() - start) * 1000)

    conversations = [_conversation(i, picks[3 * i : 3 * i + 3]) for i in range(total)]
    operator = IntentOperator(model=model)
    operator_ms = asyncio.run(_bench_operator(operator, conversations, args.warmup))

    result = {
        "model_id": model_id,
        "device": args.device,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "torch_threads": torch.get_num_threads(),
        "n": args.n,
        "utterance": percentiles(utterance_ms),
        "operator_3_turns": percentiles(operator_ms),
    }
    print(json.dumps(result, indent=2))

    if args.log_to_mlflow and args.model_uri.startswith("models:/"):
        mlflow = importlib.import_module("mlflow")
        client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        name, version = model_id.split("/v")
        run_id = client.get_model_version(name, version).tags["run_id"]
        for kind in ("utterance", "operator_3_turns"):
            for stat, value in result[kind].items():
                client.log_metric(run_id, f"latency_{args.device}_{kind}_{stat}", value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
