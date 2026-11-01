# Verifier candidate recall below τ

Input to T10. The LLM verifier re-checks intents whose calibrated confidence is below τ, and it may
only choose among the classifier's top-k candidates. Its accuracy on those cases is therefore
capped by **recall@k below τ**. This report measures that cap per k, to inform the choice of k. It
is a measurement only: no production code changed.

Reproduce (needs the `ml` group, the cached champion and the pinned data):

```bash
uv run --group ml python -m training.intent.verifier_candidates      # --device cpu|cuda
```

Config: [`../verifier_candidates.yaml`](../verifier_candidates.yaml). Raw JSON goes to `outputs/`
(gitignored).

## Setup

- Model: `intent-classifier@champion` = **v1** (run `11387b8681024f68a9a0b2df15377965`),
  T = 0.9649, **τ = 0.861**, all read from the bundle. Probabilities are
  `softmax(predict_logits, T)`, as in `IntentOperator._predict`. Confidence and candidate order
  come from `pick_intent` itself, called per utterance.
- Below τ = `confidence < τ`. The classifier serves iff `confidence >= τ`, the same boundary as
  `evaluation.point_at`.
- Data: Banking77 `mteb/banking77@18072d26`, loaded through `load_split`, which checks the sha256
  against `manifest.json`. Test = official test split (3 076). Validation = the same stratified
  10 % of train that T03 used to choose T and τ (1 002).
- Sanity check: the run reproduces the metrics logged in MLflow for v1 exactly: test accuracy
  0.926203, coverage at τ 0.875488, accuracy at or above τ 97.7 %, validation coverage 85.0 %.
- Intervals below are Wilson 95 %.

## Results

### Test (n = 3 076)

| | Value |
|---|---|
| Below τ | **383 (12.5 %)** |
| Classifier accuracy below τ (the baseline the verifier must beat) | **57.2 %** [52.2, 62.0] |
| Classifier accuracy at or above τ | 97.7 % |
| Below τ and classifier wrong | 164 |

| k | recall@k below τ | recall@k below τ, classifier wrong | max gain over baseline |
|---|---|---|---|
| 1 | 57.2 % [52.2, 62.0] | 0.0 % | 0 pp |
| 2 | 77.0 % [72.6, 81.0] | 46.3 % [38.9, 54.0] | +19.8 pp |
| 3 | 85.4 % [81.5, 88.6] | 65.9 % [58.3, 72.7] | +28.2 pp |
| 5 | **92.4 %** [89.3, 94.7] | **82.3 %** [75.8, 87.4] | +35.2 pp |
| 10 | 97.1 % [94.9, 98.4] | 93.3 % [88.4, 96.2] | +39.9 pp |

"Max gain" is recall@k minus the baseline: what a verifier would add if it always picked the
true label whenever that label is among the candidates. recall@1 equals the baseline, and recall@1
on the classifier-wrong cases is 0 by definition; both are consistency checks.

### Validation (n = 1 002): where k should be chosen

| | Value |
|---|---|
| Below τ | 150 (15.0 %) |
| Classifier accuracy below τ | 60.7 % [52.7, 68.1] |
| Classifier accuracy at or above τ | 97.1 % |
| Below τ and classifier wrong | 59 |

| k | recall@k below τ | recall@k below τ, classifier wrong |
|---|---|---|
| 1 | 60.7 % | 0.0 % |
| 2 | 80.7 % [73.6, 86.2] | 50.8 % [38.4, 63.2] |
| 3 | 90.7 % [84.9, 94.4] | 76.3 % [64.0, 85.3] |
| 5 | 93.3 % [88.2, 96.3] | 83.1 % [71.5, 90.5] |
| 10 | 96.0 % [91.5, 98.2] | 89.8 % [79.5, 95.3] |

Validation and test agree within their intervals. Validation is small (150 below τ, 59 wrong), so
its differences between neighbouring k are within noise.

### Cost: one verifier pass over the 383 below-τ test cases, claude-haiku-4-5

Priced by `convintel.llm.pricing` (1 USD in, 5 USD out per MTok).

| k | mean input tokens, utterance / conversation | USD one pass, utterance / conversation | USD per 1 000 conversations |
|---|---|---|---|
| 1 | 404 / 887 | 0.250 / 0.435 | 0.142 |
| 2 | 441 / 924 | 0.265 / 0.450 | 0.146 |
| 3 | 478 / 961 | 0.279 / 0.464 | 0.151 |
| 5 | 552 / 1 035 | 0.307 / 0.492 | 0.160 |
| 10 | 735 / 1 219 | 0.377 / 0.562 | 0.183 |

Token assumptions (estimates; no tokenizer was called):

- Fixed prompt (instructions + output schema): 350 tokens, independent of k.
- Each candidate: its label id at 2.5 chars/token (snake_case splits into many tokens), plus its
  taxonomy-v3 description at 3.5 chars/token, plus 8 tokens of list formatting. That is about
  37 tokens per candidate. The lengths are those of the actual top-k labels of each case, from
  `INTENT_DESCRIPTIONS_V3` at `origin/lane-f/T06-synthetic-data` commit `6f19aea`.
- Text being verified: the Banking77 utterance at 3.5 chars/token (~18 tokens) in the
  "utterance" scenario, or a 500-token redacted conversation in the "conversation" scenario.
- Output: 50 tokens (chosen label + one-line reason).
- "Per 1 000 conversations" = conversation scenario × the 12.5 % below-τ share.
- No prompt caching. Caching the fixed part would lower every row by roughly the same amount.

**Cost does not decide k.** Going from k = 3 to k = 10 adds about 3 cents per 1 000
conversations; the conversation text dominates the prompt.

## Recommendation (the human decides)

**k = 5.**

- It raises the ceiling where the verifier can help: on the classifier-wrong cases, test recall goes
  from 65.9 % at k = 3 to 82.3 % at k = 5 (+16 pp; the 95 % intervals do not overlap). On validation the
  same step is +7 pp, within noise but in the same direction. Below τ overall, the ceiling goes
  from 85.4 % to 92.4 % on test.
- Past 5 the returns shrink: k = 10 adds +4.7 pp on test (+2.7 on validation) while doubling the
  candidate list. Every extra candidate is a distractor that the verifier can wrongly pick,
  including when the classifier's top-1 was already right (57 % of the below-τ cases). Whether that
  trade pays is a property of the verifier, not of this table. If the T10 verifier eval shows that
  distractors cost little, k = 10 is the next step; if they cost a lot, k = 3.
- Cost is flat across k (above), and k = 5 needs no contract change: `IntentOutput.top_k` has no
  length limit and `IntentOperator(top_k=...)` already takes it. The operator default is
  `DEFAULT_TOP_K = 3`, so T10 would build the operator with `top_k=5` or change the default.
- The verifier also needs a "none of these" path: even at k = 5, the true label is missing for
  about 8 % of the below-τ cases (18 % of the classifier-wrong ones). Forcing a choice there yields
  a confident wrong label.

## Single utterances versus whole conversations

Banking77 examples are single utterances, while the pipeline verifies whole conversations. These
numbers are therefore an optimistic proxy, not the T10 ceiling:

- **Different population below τ.** `pick_intent` takes the most confident customer turn, and that
  maximum is optimistically biased. Fewer conversations fall below τ than utterances do, and those
  that still do are the genuinely ambiguous ones (several intents, vague turns, off-taxonomy
  requests). Recall@k on that population is likely lower than here.
- **Candidates come from one turn, the verifier reads them all.** The top-k list is from the single
  most confident turn. If the real request is stated in another turn, its label may be absent from
  the candidates however large k is. Pooling candidates across customer turns would close that gap;
  that is a T10 design option and was not measured here.
- **The label can be ill-defined.** A conversation can hold several intents, while Banking77 has
  exactly one per utterance. Recall@k needs a conversation-level gold label.
- **Closed set.** Banking77 has no out-of-scope class; real conversations do. This strengthens the
  case for a "none of these" answer.
- **Cost shifts toward the conversation.** With ~500 conversation tokens, the candidate list is a
  small share of the prompt, so a larger k is relatively cheaper than in the utterance scenario.

Before k is frozen, the same measurement should be repeated at conversation level on the T06
golden set, together with τ (which the T03 README already flags for re-tuning there).

## Mutation check on `recall_at_k`

The prediction was written in the body of commit `fce4d54` before running. Each mutation was
applied to the slice `c[:k]`, the unit tests were run, and the file was restored with
`git checkout`.

| Mutation | Predicted | Observed |
|---|---|---|
| `c[: k + 1]` | 4 failures: parametrized k = 1, 2; rank-k test; study_split test | **4 failed, 13 passed**, exactly those |
| `c[: k - 1]` | 5 failures: parametrized k = 1, 2, 3; rank-k test; study_split test | **5 failed, 12 passed**, exactly those |

After restoring the file, the 17 tests pass.
