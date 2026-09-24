# T03 — Intent classifier (fine-tuning + registry)

**Lane:** C · **Depends on:** T00 (T06 optional: extra synthetic training data) · **Bake-off task: Pi vs Claude Code**

## Goal
A fast, calibrated intent classifier that knows when it is unsure, so that the pipeline can
route low-confidence cases to an LLM verifier (ADR-0001).

## Allowed paths
`src/convintel/operators/intent/`, `training/intent/`, `tests/operators/intent/`

## Requirements
- Training script `training/intent/train.py`: fine-tune `distilbert-base-uncased` on Banking77
  (HF `datasets`), with reproducible seeds and a config file.
- Log params, metrics and the model to **MLflow**; register it as `intent-classifier` and tag it with
  the dataset version.
- **Calibration:** temperature scaling on the validation set. Report ECE before and after.
- Threshold analysis: coverage vs accuracy curve. Pick τ so that accuracy ≥ 97 % on the covered
  cases, and report the resulting coverage.
- `IntentOperator` implementing `Operator[IntentOutput]`: loads the model by registry alias
  (`@champion`), classifies the customer turns, returns `top_k` and calibrated `confidence`,
  `source="classifier"`. ONNX export optional (bonus: latency comparison).

## Acceptance criteria
- [ ] Macro-F1 on the Banking77 test split reported (target ≥ 0.90).
- [ ] ECE before/after calibration, and the coverage/accuracy curve saved as MLflow artifacts.
- [ ] The operator's unit tests use a tiny stub model; the integration test loads from MLflow.
- [ ] p95 inference latency on CPU reported in the PR.

## Out of scope
LLM verifier (T10), serving (T12).
