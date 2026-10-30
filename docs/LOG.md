# Engineering log

Dated notes: decisions, agent behaviour, surprises, numbers. Feeds the README's "How I built this".

## 2026-09-23
- Defined architecture, contracts, task briefs (T00–T06) and multi-agent workflow.
- Next: T00 scaffold, then freeze contracts and open Phase 1 issues.

## 2026-10-30
- **#22 (closes #21): Ollama retries.** The issue said Ollama did not retry 5xx or dropped
  connections. Reading the installed libraries first showed a third gap: `ChatOllama` always
  streams, and on the streaming path even a refused connection arrives as a raw
  `httpx.ConnectError`, so nothing was retried. Fix: 5 lines in `_is_retryable`. Tests drive the
  real `ChatOllama` through `httpx.MockTransport` (no network), so the library raises its own
  exceptions instead of the ones we assume. An in-band stream error (`status_code=-1` after a
  200) stays non-transient, now pinned by a test.
- **#24: agent rules in AGENTS.md.** The independent review caught that rules describing the
  pipeline (not yet on `main`) were written in the present tense, so an agent working on `main`
  would skip instrumentation. They are now marked *Target contract (T10)* with a note on the
  current state. The author traced two wrong line references in #23 to their cause (line
  numbers taken from concatenated output) and found three more.
- **T10 (draft PR #25): spans, warmup and the factory.**
  - Sentiment and summary still opened their own `operator.<name>` span, which would have
    nested a duplicate under the pipeline's `traced()` span. Moving them to the caller's span
    broke no summary test, because there was none for its spans: losing token and cost
    attributes would have passed silently. Added the test; it fails on the old code.
  - `build_pipeline(settings)` is cheap and side-effect free; all I/O moved to `warmup()`.
    PII and sentiment used to load their models on the first request, so the service reported
    ready before it was. Now a missing model fails startup, not the first customer.
  - The summary warmup validates the LLM config without network (a missing key is a permanent
    misconfiguration, so it fails startup; fail-open stays for transient provider errors). No
    LLM call at startup: that would tie our availability to the provider's.
  - Warmups run concurrently. Review found no timeout (the intent warmup downloads from MLflow,
    which retries for minutes) and that `gather` reports only the first failure. Next:
    `asyncio.TaskGroup` plus `asyncio.timeout`.
  - A hidden global dependency (`get_settings()` inside the intent model loader) cannot diverge
    in production; logged in #20 instead of widening the PR.
- **T12 part 2 design** (read-only agent): Cloud Tasks plus a private worker service, job
  state in Postgres. It rejected in-process background tasks because Cloud Run throttles CPU
  after the response is sent. 14 decisions pending review.
- **Process:** shared state moved to GitHub. Agents push every commit to a draft PR, write each
  mutation's prediction and result in the commit body, and post their final report as a PR
  comment; the reviewer posts its review on the PR. Only squash merges are allowed now.
- Next: integration test with real models (`-m integration`), then the LLM intent verifier.
