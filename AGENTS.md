# AGENTS.md

Instructions for every coding agent working in this repo (Claude Code, Devin, Pi, Cursor…).
This is the single source of truth. Do **not** create a `CLAUDE.md` — Claude Code reads this file
when no `CLAUDE.md` exists.

## What this project is

A production-grade **Conversation Intelligence Engine**: it ingests voice calls and chat
conversations, transcribes them, redacts PII, runs analysis operators (sentiment, intent,
summary, structured extraction) orchestrated with LangGraph, and serves results through a
FastAPI service instrumented with OpenTelemetry, Prometheus and Arize Phoenix.

The goal is not "a demo that works on my laptop". Every feature must be **tested, traced,
measured and deployable**. Read `docs/ARCHITECTURE.md` before touching code.

## Commands

```bash
uv sync                          # install deps
uv run pytest -q                 # unit tests (must pass before any commit)
uv run pytest -m integration     # integration tests (need docker compose up)
uv run ruff check . --fix && uv run ruff format .
uv run pyright                   # type check (strict on src/convintel/contracts)
uv run python -m evals.run --suite smoke   # quick LLM eval (needs API key)
docker compose -f infra/docker-compose.yml up -d   # Phoenix, Prometheus, Grafana, MLflow
uv run uvicorn convintel.api.main:app --reload
```

pyright ships its own Node (`pyright[nodejs]`). On Node or `libatomic` errors, run `uv sync`;
never install Node or system packages.

The Docker stack (`infra/docker-compose.yml`) is always started from the main checkout, never
from a worktree.

## Repo map and ownership

| Path | Owner | Notes |
|---|---|---|
| `src/convintel/contracts/` | **Human only** | Pydantic schemas + operator protocol. Frozen. |
| `src/convintel/ingest/`, `transcription/` | Lane A | Twilio webhook, files, faster-whisper |
| `src/convintel/operators/pii/`, `sentiment/` | Lane B | Presidio, small HF model |
| `src/convintel/operators/intent/` | Lane C | Fine-tuned DistilBERT + MLflow registry |
| `src/convintel/operators/llm/`, `prompts/` | Lane D | Summary + extraction via LLM |
| `infra/`, `src/convintel/telemetry/` | Lane E | OTel, Prometheus, Grafana, Phoenix |
| `evals/`, `data/synthetic/` | Lane F | Synthetic data + golden sets + eval runner |
| `src/convintel/graph/`, `api/` | Phase 2 | Integration — only after lanes merge |

**Stay in your lane.** Only edit the paths listed in your task file (`docs/tasks/Txx-*.md`).
If you need something outside your lane, stop and write it in the PR description under
"Requests for other lanes" instead of editing it yourself. The only exception is explicit human
authorization: then make the change in a separate commit and note it in the PR as
"cross-lane change approved by the human".

## Hard rules

1. **Contracts are frozen.** Never edit `src/convintel/contracts/`. Code against them.
   If a contract is wrong, say so in the PR; the human decides.
2. **Every operator implements `Operator`** from `contracts/operators.py` and returns an
   `OperatorResult` (with `latency_ms`, `model_id`, and `cost_usd` / `prompt_version` when applicable).
3. **No inline prompts.** Prompts live in `prompts/<name>/v<N>.yaml` and are immutable once
   merged (ADR-0004) or once used to produce data or results, whichever comes first. Changing a
   prompt means adding a new version file, never editing an old one.
4. **All LLM calls go through `convintel.llm.client`** (tracing, retries, token and cost accounting).
   Never instantiate a provider SDK directly in an operator. Model-dependent facts (price, whether
   sampling parameters are accepted) live in tables in the LLM client, never in prompts.
5. **Never log raw PII.** Log conversation IDs and redacted text only. Transcripts in logs,
   traces or fixtures must be redacted or synthetic.
6. **No secrets in code.** Read config from env via `convintel.config.Settings`. Add new
   variables to `.env.example` (a test enforces it). In tests, build `Settings(_env_file=None)`
   and call `get_settings.cache_clear()` so a local `.env` never leaks into assertions.
   Never read or print `.env`.
7. **New dependencies need a reason.** Add with `uv add`, and justify it in the PR.
8. **Tests are part of the task.** New code ships with unit tests. Mock external services
   (LLM APIs, Twilio, model downloads) in unit tests. Put anything that hits the network behind `@pytest.mark.integration`.
9. **Instrument what you build.** New code paths emit an OTel span and, if user-facing,
   a Prometheus metric following the naming in `docs/ARCHITECTURE.md#telemetry`.
   *Target contract (T10):* operators do not instrument themselves: they annotate the current
   span, and the pipeline wraps every operator call with `traced()` (spans + latency/error metrics).
   Phoenix groups traces by `openinference.project.name`.
10. **Never relax ruff, pyright or pytest configuration to make checks pass.** Fix the code or
    report it in the PR.
11. **The `ml` group is for work with local models** (`uv sync --group ml`); torch/CUDA go only
    there, never in base deps or `dev`. Code that needs them must skip cleanly without them
    (`pytest.importorskip` + the `gpu` marker) so CI stays green. Never download models from
    external hubs at runtime; models come from the local cache or our MLflow registry.
    *Target contract (T10):* models are loaded in `warmup()`.

*Target contract (T10)* marks rules that land with T10 (branch `lane-g/T10-graph`). Until T10 is
merged, some operators still open their own spans and some models load lazily on first use
(tracked in #23); do not add new self-instrumentation or lazy loading.

## Code style

- Python 3.12, fully typed, `async` for I/O-bound code.
- Pydantic v2 models for all data crossing module boundaries. No bare dicts.
- Small pure functions; side effects at the edges (API, ingest, storage).
- Docstrings explain *why*, not *what*. No commented-out code.
- Errors: raise domain exceptions from `convintel.errors`; never swallow exceptions silently.
  When the original exception could contain conversation text, the new error carries only
  `type(exc).__name__` and is raised with `raise ... from None`.
- *Target contract (T10):* the pipeline only catches `ConvIntelError`: an expected failure
  degrades to a `None` result plus `"<operator>:<ErrorType>"`. Any other exception is a bug
  and propagates.

## Git and PRs

- Branch: `lane-<x>/<task-id>-<slug>` (e.g. `lane-c/T03-intent-classifier`).
- Conventional commits: `feat(intent): add confidence calibration`.
- One task = one PR. Keep PRs under ~400 changed lines when possible.
- Fill in `.github/pull_request_template.md` completely, including how you tested.
- Do not merge. The human reviews and merges every PR.

## Definition of done

- [ ] Acceptance criteria in the task file are met
- [ ] `ruff`, `pyright`, `pytest` pass locally and in CI
- [ ] New code is traced (span) and measured (metric) where relevant
- [ ] Docs updated (module README or `docs/`) if behaviour or interfaces changed
- [ ] If a technical decision was made, propose it in the PR; ADRs in `docs/adr/` are numbered
      and created by the human

## When you are unsure

Prefer the simplest thing that satisfies the contract and the acceptance criteria.
Do not invent requirements. If the task is ambiguous, list your assumptions at the top of the
PR description instead of guessing silently.
