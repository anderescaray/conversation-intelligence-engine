# T00 — Scaffold, tooling and CI

**Lane:** — (pair with the human) · **Depends on:** nothing · **Status:** ✅ done (PRs #1, #2)

## Goal
A repo where every later task can start immediately: installable package, lint, types, tests, CI.

## Allowed paths
`pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `.github/workflows/`, `.env.example`,
`src/convintel/__init__.py`, `src/convintel/config.py`, `src/convintel/errors.py`, `tests/`,
`.devcontainer/`

## Requirements
- `uv` project, Python 3.12, package `convintel` in `src/` layout.
- Dev deps: ruff, pyright, pytest, pytest-asyncio, pre-commit.
- Ruff config (lint + format), pyright `strict` for `src/convintel/contracts`, `basic` elsewhere.
- `convintel.config.Settings` (pydantic-settings) reading env: LLM provider/model, API keys, Phoenix endpoint, MLflow URI, Postgres URL, Twilio creds.
- `convintel.errors`: base `ConvIntelError` and `OperatorError`, `TranscriptionError`, `LLMError`.
- GitHub Actions: on PR → ruff, pyright, pytest (unit only). Cache uv.
- Devcontainer (Python 3.12 + Node for Pi) so agents run sandboxed.
- pytest marker `integration` registered and excluded by default.

## Acceptance criteria
- [ ] `uv sync && uv run pytest` passes on a clean clone.
- [ ] A test imports `convintel.contracts` and builds a sample `Conversation`.
- [ ] CI runs and passes on a PR.
- [ ] `.env.example` lists every setting with a comment.

## Out of scope
Any operator, API or infra code.
