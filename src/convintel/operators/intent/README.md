# Intent (`convintel.operators.intent`)

`IntentOperator` implements `Operator[IntentOutput]` on a **redacted** conversation: a
DistilBERT fine-tuned on Banking77 (77 banking intents), with temperature-scaled confidences so
that the pipeline can route low-confidence cases to the LLM verifier (ADR-0001).

```python
from convintel.operators.intent import IntentOperator

operator = IntentOperator()          # models:/intent-classifier@champion
await operator.warmup()              # at startup: resolve, load, one dummy inference
result = await operator.run(conversation)
result.output        # IntentOutput(label, confidence, source="classifier", top_k=[(label, p), ...])
operator.threshold   # τ shipped with the loaded model version
```

Training, calibration and the threshold study live in [`training/intent/`](../../../../training/intent/README.md).

## Behaviour

- Only customer turns are classified, in one batch. Banking77 is single-utterance, so each turn
  is scored on its own and the turn with the **highest calibrated top-1 confidence** wins; its
  distribution gives `top_k` (default 3). Greetings and confirmations score low and lose.
- `confidence` is `softmax(logits / T)`, with T fitted on the validation split and stored in the
  model bundle, so a model version is never served with another version's temperature.
- No customer turns → `label="unknown"`, `confidence=0.0`, `top_k=[]`, model not loaded,
  `model_id=None`. There is nothing to verify: the pipeline must **not** send these to the LLM
  verifier (requested from T10).
- The operator does not route. τ is exposed as `operator.threshold` for the graph to use.

### Known bias: max over turns

Each turn's confidence is calibrated; their **maximum is not**. Taking the most confident of
several turns is optimistically biased (selection effect), more so the more customer turns a
conversation has. τ was chosen per utterance on Banking77 validation; it must be **re-tuned at
conversation level on the T06 golden set** before the pipeline routes on it.

## Model resolution

`model_uri` accepts:

| Form | Example | Needs |
|---|---|---|
| Registry alias | `models:/intent-classifier@champion` (default) | MLflow server, `mlflow` |
| Registry version | `models:/intent-classifier/3` | MLflow server, `mlflow` |
| Local bundle directory | `/srv/models/intent/bundle` | nothing but torch/transformers |

- A registry URI is resolved against `MLFLOW_TRACKING_URI` (`Settings.mlflow_tracking_uri`,
  the project's own MLflow server) and the version is downloaded **once** to
  `~/.cache/convintel/models/<server-hash>/<name>/v<version>/` (atomic rename, so an
  interrupted download never leaves a half cache). Later starts read the cache.
- The alias is resolved once per process: promoting a new champion takes effect on restart.
- Nothing is ever fetched from the Hugging Face hub at runtime (`local_files_only=True`): the
  tokenizer and weights come from the bundle.
- A bundle is a `save_pretrained` directory plus `intent_meta.json` (labels in logit order,
  T, τ, `max_length`, base model, dataset version).

## Runtime

- The model is a lazy, lock-protected per-process singleton (`get_intent_model`). Failures are
  not cached, so a later call retries (e.g. once MLflow is up).
- Inference runs in a worker thread under a process-wide lock, one batch at a time per process:
  **scale with worker processes**, not threads. Device: CUDA if torch sees it, else CPU
  (override with `device=`).
- Needs `torch` and `transformers` (plus `mlflow` for registry URIs), all in the `ml` group
  (ADR-0005), imported only when the model loads. The production image must install them (T13).

## Errors

- Missing bundle, unreachable registry, unknown alias, missing `ml` group:
  `OperatorError("intent", "<explicit message>")`. These messages are built from paths and URIs
  only; registry errors keep the exception type, not the server's message.
- Anything else: `OperatorError("intent", "intent inference failed (<ExceptionType>)")`, with
  no chained cause, so text quoted in third-party messages never reaches spans or logs (same
  policy as the PII and sentiment operators).

## Telemetry

- `operator.intent` span via `traced`, which also records `convintel_operator_latency_seconds`
  and `convintel_operator_errors_total` (`operator="intent"`). Attributes: `conversation.id`,
  `operator.version`, `model.id`, `intent.customer_turns`, `intent.label`, `intent.confidence`.
  Never text.
- `operator.intent.warmup` span on `warmup()` (no metrics, so startup does not skew latency).

## Tests

- Unit: `FakeIntentModel` (canned logits, no torch, no network); calibration, threshold study,
  registry resolution with a fake MLflow, sanitized errors, spans, warmup, lock.
- `test_intent_hf.py`: real transformers adapter on a tiny random DistilBERT built in the test
  (`ml` group, no network; skipped in CI).
- Integration: loads `@champion` from the compose MLflow. `-m "integration and not gpu"` runs
  the CPU case only; `-m integration` also runs the CUDA case (skipped without a GPU).
