"""Fine-tune DistilBERT on Banking77, calibrate it, study τ, then log and register in MLflow.

    uv run --group ml python -m training.intent.train [--config PATH] [--no-promote]

Order of operations, which is what keeps the test split untouched until the end:
1. train on the stratified 90 % of the official train split, pick the best epoch by
   validation macro-F1;
2. fit the temperature and choose τ on validation (`report.calibration_study`);
3. save the bundle (weights, tokenizer, T, τ) and freeze it;
4. `final_evaluation`: the only function that loads the test split, run once on the saved bundle.
The champion alias is then moved only if test macro-F1 reaches the T03 target (a pass/fail
gate on a frozen model, not a selection among candidates).
"""

import argparse
import csv
import importlib
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from convintel.config import get_settings
from convintel.operators.intent import IntentBundleMeta, load_bundle
from convintel.operators.intent.bundle import write_bundle_meta
from convintel.operators.intent.calibration import negative_log_likelihood
from convintel.operators.intent.evaluation import accuracy, macro_f1
from convintel.operators.intent.model import BUNDLE_ARTIFACT, CHAMPION_ALIAS

from .config import DEFAULT_CONFIG, TrainConfig, load_config
from .data import LabeledTexts, load_split, read_manifest, stratified_split
from .plots import coverage_accuracy_plot, reliability_diagram
from .report import CalibrationStudy, FrozenEvaluation, calibration_study, evaluate_frozen

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PYFUNC_FILE = Path(__file__).resolve().parent / "pyfunc_model.py"
LOGGED_MODEL_NAME = "intent_model"


def set_seed(seed: int, torch: ModuleType) -> None:
    """Seed every RNG in play. Bit-exact GPU reruns are still not guaranteed (atomics in
    some CUDA kernels); the metrics are reproducible to within noise."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _encode(tokenizer: Any, texts: list[str], cfg: TrainConfig, device: str) -> Any:
    return tokenizer(
        texts, padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt"
    ).to(device)


def predict_logits(
    model: Any, tokenizer: Any, texts: tuple[str, ...], cfg: TrainConfig, device: str
) -> NDArray[np.float32]:
    """fp32 logits, like the operator at serving time, so validation numbers match serving."""
    torch = importlib.import_module("torch")
    model.eval()
    out: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for start in range(0, len(texts), cfg.eval_batch_size):
            batch = list(texts[start : start + cfg.eval_batch_size])
            out.append(model(**_encode(tokenizer, batch, cfg, device)).logits.float().cpu().numpy())
    return np.concatenate(out)


def train_epochs(
    model: Any,
    tokenizer: Any,
    train: LabeledTexts,
    val: LabeledTexts,
    cfg: TrainConfig,
    device: str,
) -> int:
    """Train in place and leave the best epoch's weights (by validation macro-F1) loaded."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    mlflow = importlib.import_module("mlflow")

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
    steps_per_epoch = math.ceil(len(train.texts) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps
    )
    use_bf16 = cfg.bf16 and device.startswith("cuda")
    generator = torch.Generator().manual_seed(cfg.seed)
    labels = torch.tensor(train.labels)

    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(train.texts), generator=generator).tolist()
        running = 0.0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            encoded = _encode(tokenizer, [train.texts[i] for i in idx], cfg, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**encoded, labels=labels[idx].to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += loss.item()

        val_logits = predict_logits(model, tokenizer, val.texts, cfg, device)
        predictions = val_logits.argmax(axis=1)
        val_f1 = macro_f1(predictions, val.labels, len(val.label_names))
        mlflow.log_metrics(
            {
                "train_loss": running / steps_per_epoch,
                "val_loss": negative_log_likelihood(val_logits, val.labels, 1.0),
                "val_accuracy": accuracy(predictions, val.labels),
                "val_macro_f1": val_f1,
            },
            step=epoch,
        )
        print(f"epoch {epoch}: train_loss={running / steps_per_epoch:.4f} val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return best_epoch


def final_evaluation(bundle_dir: Path, cfg: TrainConfig, device: str) -> FrozenEvaluation:
    """The only reader of the test split. Scores the saved bundle (weights, T and τ) exactly
    as the operator loads it, so the reported numbers are those of the registered artifact."""
    test = load_split("test")
    model = load_bundle(bundle_dir, model_id="final-evaluation", device=device)
    logits = model.predict_logits(test.texts)
    return evaluate_frozen(
        logits,
        test.labels,
        temperature=model.temperature,
        threshold=model.threshold,
        n_bins=cfg.ece_bins,
        num_classes=len(test.label_names),
    )


def write_reports(
    report_dir: Path, cfg: TrainConfig, study: CalibrationStudy, test: FrozenEvaluation
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"validation": study.model_dump(), "test": test.model_dump()}
    (report_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (report_dir / "coverage_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["threshold", "val_coverage", "val_accuracy", "test_coverage", "test_accuracy"]
        )
        for v, t in zip(study.curve, test.curve, strict=True):
            writer.writerow([v.threshold, v.coverage, v.accuracy, t.coverage, t.accuracy])
    reliability_diagram(
        study.bins_before,
        study.bins_after,
        f"Validation reliability (T = {study.temperature:.3f})",
        report_dir / "reliability_validation.png",
    )
    reliability_diagram(
        test.bins_before,
        test.bins_after,
        f"Test reliability (T = {study.temperature:.3f})",
        report_dir / "reliability_test.png",
    )
    coverage_accuracy_plot(
        {"validation": study.curve, "test": test.curve},
        study.threshold.threshold if study.threshold else None,
        cfg.target_covered_accuracy,
        report_dir / "coverage_accuracy.png",
    )


def summary_metrics(study: CalibrationStudy, test: FrozenEvaluation) -> dict[str, float]:
    metrics = {
        "temperature": study.temperature,
        "val_ece_before": study.ece_before,
        "val_ece_after": study.ece_after,
        "val_nll_before": study.nll_before,
        "val_nll_after": study.nll_after,
        "test_accuracy": test.accuracy,
        "test_macro_f1": test.macro_f1,
        "test_ece_before": test.ece_before,
        "test_ece_after": test.ece_after,
    }
    if study.threshold is not None and study.threshold.accuracy is not None:
        metrics |= {
            "tau": study.threshold.threshold,
            "val_coverage_at_tau": study.threshold.coverage,
            "val_covered_accuracy_at_tau": study.threshold.accuracy,
        }
    if test.at_threshold is not None:
        metrics["test_coverage_at_tau"] = test.at_threshold.coverage
        if test.at_threshold.accuracy is not None:
            metrics["test_covered_accuracy_at_tau"] = test.at_threshold.accuracy
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune, calibrate and register the intent classifier."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--no-promote", action="store_true", help="never move @champion")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    manifest = read_manifest()
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    mlflow = importlib.import_module("mlflow")
    set_seed(cfg.seed, torch)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    full = load_split("train")
    train_idx, val_idx = stratified_split(full.labels, cfg.val_fraction, cfg.seed)
    train, val = full.subset(train_idx), full.subset(val_idx)
    names = full.label_names

    # The base checkpoint is fetched from the Hugging Face hub here, at training time only.
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.base_model)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={name: i for i, name in enumerate(names)},
    ).to(device)

    run_dir = args.output_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bundle_dir = run_dir / BUNDLE_ARTIFACT
    tracking_uri = get_settings().mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(cfg.model_dump())
        mlflow.set_tags(
            {
                "dataset.version": manifest.version,
                "dataset.train_size": len(train.texts),
                "dataset.val_size": len(val.texts),
                "device": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
                "torch.version": torch.__version__,
            }
        )
        mlflow.log_artifact(str(args.config))

        started = time.perf_counter()
        best_epoch = train_epochs(model, tokenizer, train, val, cfg, device)
        train_seconds = time.perf_counter() - started
        mlflow.log_metrics({"train_seconds": train_seconds, "best_epoch": best_epoch})

        study = calibration_study(
            predict_logits(model, tokenizer, val.texts, cfg, device),
            val.labels,
            cfg.ece_bins,
            cfg.target_covered_accuracy,
            cfg.min_covered_cases,
        )
        model.save_pretrained(bundle_dir)
        tokenizer.save_pretrained(bundle_dir)
        meta = IntentBundleMeta(
            labels=names,
            temperature=study.temperature,
            threshold=study.threshold.threshold if study.threshold else None,
            max_length=cfg.max_length,
            base_model=cfg.base_model,
            dataset_version=manifest.version,
        )
        write_bundle_meta(bundle_dir, meta)

        test = final_evaluation(bundle_dir, cfg, device)
        metrics = summary_metrics(study, test)
        mlflow.log_metrics(metrics)
        write_reports(run_dir / "report", cfg, study, test)
        mlflow.log_artifacts(str(run_dir / "report"), artifact_path="report")

        info = mlflow.pyfunc.log_model(
            name=LOGGED_MODEL_NAME,
            python_model=str(PYFUNC_FILE),
            artifacts={BUNDLE_ARTIFACT: str(bundle_dir)},
            pip_requirements=[
                f"torch=={torch.__version__.split('+')[0]}",
                f"transformers=={transformers.__version__}",
                f"mlflow=={mlflow.__version__}",
            ],
            metadata=meta.model_dump(mode="json"),
            registered_model_name=cfg.registered_model,
        )
        version = str(info.registered_model_version)
        client = mlflow.MlflowClient()
        for key, value in {
            "dataset.version": manifest.version,
            "test_macro_f1": f"{test.macro_f1:.4f}",
            "temperature": f"{study.temperature:.4f}",
            "tau": "none" if meta.threshold is None else f"{meta.threshold:.3f}",
            "run_id": run.info.run_id,
        }.items():
            client.set_model_version_tag(cfg.registered_model, version, key, value)

        promoted = not args.no_promote and test.macro_f1 >= cfg.promote_min_macro_f1
        if promoted:
            client.set_registered_model_alias(cfg.registered_model, CHAMPION_ALIAS, version)
        mlflow.set_tag("promoted", str(promoted).lower())

    print(json.dumps(metrics | {"version": version, "promoted": promoted}, indent=2))
    print(f"bundle: {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
