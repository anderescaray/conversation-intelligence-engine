# Intent classifier training (T03)

Fine-tunes `distilbert-base-uncased` on Banking77, calibrates it with temperature scaling,
chooses the routing threshold τ, and registers the result in MLflow as `intent-classifier`.
The operator that serves it is [`convintel.operators.intent`](../../src/convintel/operators/intent/README.md).

Everything here needs the `ml` group: `uv sync --group ml`.

## 1. Data (one-time, explicit)

```bash
uv run python -m training.intent.download_data
```

- Source: [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) at commit
  `18072d2685ea682290f7b8924d94c62acc19c0b2`. The original `PolyAI/banking77` repo only ships a
  loading script, which `datasets` no longer runs; the MTEB copy is the same split as parquet.
  Compared with the original PolyAI CSVs it lacks 10 of 10 003 train and 4 of 3 080 test
  utterances, so it has **9 993 train / 3 076 test**. Numbers are comparable with the literature
  to within that 0.1 %.
- Files land in `training/intent/data/` (gitignored) with a `manifest.json` holding the sha256
  of each file. Training refuses files that do not match. `dataset.version` in MLflow is
  `mteb/banking77@<commit>`.
- This is the only step that downloads data. Training reads disk only; unit tests never touch
  the data or the network.

## 2. Splits, and keeping test untouched

| Split | Size | Used for |
|---|---|---|
| train | 90 % of official train, stratified, seed 42 (8 991) | gradient updates |
| validation | 10 % of official train, stratified (1 002, all 77 classes) | best epoch, T, τ |
| test | official test (3 076) | **final evaluation only** |

`train.final_evaluation` is the only function that loads the test split (a unit test checks
this on the source). It runs once, on the bundle already saved to disk with T and τ frozen,
loaded exactly as the operator loads it. The champion gate (test macro-F1 ≥ 0.90) is a
pass/fail check on that frozen model, not a selection among candidates.

## 3. Training

```bash
docker compose -f infra/docker-compose.yml up -d mlflow     # tracking + registry
uv run python -m training.intent.train                      # --no-promote, --device cpu
```

Config: [`config.yaml`](config.yaml) (validated by `TrainConfig`, unknown keys rejected).

| Setting | Value | Why |
|---|---|---|
| base model | `distilbert-base-uncased` | ADR-0001: fast, cheap, good on a closed label set |
| max_length | 128 | longest train utterance is 98 tokens; padding is per batch, so free |
| batch / lr | 32 / 5e-5, AdamW, wd 0.01 (not on bias/LayerNorm) | standard BERT fine-tuning |
| schedule | linear, 10 % warmup, 5 epochs, grad clip 1.0 | |
| precision | bf16 autocast on CUDA (fp32 master weights); evaluation in fp32 | matches serving |
| seed | 42 for python, numpy, torch, shuffling and the split | reproducible to within GPU nondeterminism |

Plain PyTorch loop (no HF `Trainer`): no `accelerate` dependency, and explicit control over
which split each phase sees. On an RTX 5060 Laptop (8 GB) the 5 epochs take ~85 s.

Outputs go to `training/intent/outputs/<UTC timestamp>/` (gitignored) and to MLflow:

- params (the whole config) and tags `dataset.version`, split sizes, device, torch version;
- per-epoch `train_loss`, `val_loss`, `val_accuracy`, `val_macro_f1`;
- `temperature`, `val_ece_before/after`, `val_nll_before/after`, `tau`,
  `val_coverage_at_tau`, `test_accuracy`, `test_macro_f1`, `test_ece_before/after`,
  `test_coverage_at_tau`, `test_covered_accuracy_at_tau`;
- artifacts under `report/`: `reliability_validation.png`, `reliability_test.png`,
  `coverage_accuracy.png`, `coverage_accuracy.csv`, `report.json`;
- the model: a pyfunc wrapper ([`pyfunc_model.py`](pyfunc_model.py)) whose `bundle` artifact is
  the `save_pretrained` directory plus `intent_meta.json` (T, τ, labels). Registered as a new
  version of `intent-classifier`, tagged `dataset.version`, `test_macro_f1`, `temperature`,
  `tau`, `run_id`. Alias `champion` moves to it only if test macro-F1 ≥ `promote_min_macro_f1`.

The MLflow `transformers` flavor was not used: it requires `torchvision` for this model and
the operator would depend on its internal layout. The pyfunc wrapper keeps the registry entry
directly usable with `mlflow.pyfunc.load_model`, while the operator reads only the bundle.

## 4. Calibration and τ

- Temperature: one scalar minimizing validation NLL (golden-section search on log T). It
  leaves the argmax, hence accuracy and macro-F1, unchanged.
- ECE: 15 equal-width bins, before and after, on validation and test, with reliability diagrams.
- τ: on the validation coverage/accuracy curve (thresholds every 0.001), the highest-coverage
  threshold whose covered accuracy is ≥ 97 % and that serves at least 50 cases (below that,
  one error moves covered accuracy by more than 2 points). Then reported on test.
- τ is per utterance. The operator takes the most confident customer turn, which is
  optimistically biased; τ must be re-tuned at conversation level on the T06 golden set.

## 5. Latency

```bash
uv run python -m training.intent.bench_latency --device cpu --log-to-mlflow
uv run python -m training.intent.bench_latency --device cuda --log-to-mlflow
```

p50/p95/p99 for one utterance (batch 1) and for `IntentOperator.run` on a conversation with three
customer turns, after warm-up, on random test utterances (used as text only). `--log-to-mlflow`
adds `latency_<device>_<kind>_<stat>` metrics to the training run of the resolved version.

## Results

First run: `intent-classifier` v1 (`@champion`), MLflow run `11387b8681024f68a9a0b2df15377965`,
RTX 5060 Laptop, bf16, best epoch 5 of 5, 83 s of training.

| Metric | Validation (1 002) | Test (3 076) |
|---|---|---|
| Accuracy | | **0.926** |
| Macro-F1 | 0.920 (best epoch) | **0.926** (target ≥ 0.90) |
| ECE, T = 1 → T = 0.965 | 0.0215 → 0.0213 | 0.0195 → 0.0174 |
| NLL, T = 1 → T = 0.965 | 0.3378 → 0.3370 | |
| τ = 0.861: coverage / covered accuracy | 85.0 % / 97.1 % | **87.5 % / 97.7 %** |

The fine-tuned model is already close to calibrated (fitted T ≈ 0.97, slightly *under*-confident),
so temperature scaling changes little here; it is kept because it is free, and the next model
or dataset may not be so well behaved. With τ = 0.861, about 1 conversation in 8 would go to the
LLM verifier (per utterance; see the selection-bias note above).

Latency (500 timed samples after 20 warm-up, batch 1, 16 torch CPU threads):

| | CPU p50 | CPU p95 | GPU p50 | GPU p95 |
|---|---|---|---|---|
| One utterance | 14.9 ms | 19.8 ms | 3.0 ms | 4.7 ms |
| `IntentOperator.run`, 3 customer turns | 23.1 ms | 34.4 ms | 5.4 ms | 9.3 ms |
