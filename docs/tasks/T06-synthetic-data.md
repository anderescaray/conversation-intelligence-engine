# T06 — Synthetic conversations and golden set

**Lane:** F · **Depends on:** T00, T04 (LLM client)

## Goal
Realistic labelled customer-service conversations (text and audio) with **known ground truth**.
They serve as eval data and as extra training data. This removes the dependency on private call data.

## Allowed paths
`data/synthetic/`, `evals/datasets/`, `scripts/synth/`, `tests/synth/`

## Requirements
- Generator: a scenario spec (intent from the Banking77 label set, customer mood trajectory, resolved yes/no,
  PII to include, length) goes to an LLM, which writes the dialogue and returns a JSON `Conversation` plus labels.
  Every LLM call goes through `convintel.llm.client` (T04) with a versioned prompt in `prompts/`; no provider SDKs.
- Diversity controls: personas, verbosity, ASR-like noise (disfluencies, typos in chat).
- Audio: TTS with two distinct voices, one per speaker, rendered as a dual-channel WAV (mirrors Twilio recordings).
- Golden set v1: 150 conversations (100 chat, 50 voice), stratified by intent, **with labels reviewed by the human**.
  Stored as JSONL + audio, versioned (DVC or git-LFS: justify the choice).
- Datasheet `evals/datasets/README.md`: how the set was generated, label schema, known biases.

## Acceptance criteria
- [ ] `uv run python -m scripts.synth.generate --n 10` produces valid `Conversation` JSON (validated by the contracts).
- [ ] Label distribution report generated.
- [ ] No real personal data. All PII is synthetic and flagged in the labels (for T02 recall tests).

## Out of scope
Eval runner and CI gate (T11).
