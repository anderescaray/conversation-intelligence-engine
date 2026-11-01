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

## 2026-11-01
- **T12 part 1 merged as a 4-PR stack (#27-#30).** The ~1,900-line branch was split by topic
  (spool, app and lifespan, conversations endpoint, Twilio webhooks) with no behaviour change.
  Proof: the diff between the top of the stack and the original branch is empty, and each branch
  passes on its own with `uv run --exact`. Four independent reviews, all "approve with nits"
  (#31). The most useful finding: the audio duration check used `duration > max`, which lets NaN
  through because every comparison with NaN is false.
- **T10 pipeline merged without the LLM verifier, as a 5-PR stack (#36-#40).** It was the first
  independent review of this code. The only content change allowed in the split was the task
  brief, brought in line with what was built. Two acceptance items it still lists are not met yet
  (effective τ on a span, a graph-level parallelism test); they stay in the brief and move to the
  verifier PR (#41) instead of rewriting the brief to match the code.
- **End-to-end integration test** with real models and the production summary path
  (claude-haiku-4-5): `analyze()` takes 2.1-2.6 s, almost all of it the summary call; local
  operators stay under 0.3 s; about $0.002 per conversation. A second test plants synthetic PII
  and asserts that no value reaches any exported span with the production instrumentation on. It
  also asserts that redacted placeholders do appear, so it cannot pass vacuously. Mutation: feeding
  the unredacted conversation into the graph leaks 6 values and fails the test.
- **Warmup:** `asyncio.TaskGroup` instead of `gather`, so every failed warmup is reported, plus an
  overall `asyncio.timeout`, so a hung model load fails startup instead of leaving the service up
  and never ready.
- **Intent descriptions shared and versioned (#32).** One read-only module, no default version,
  sha256 fingerprints per version. Making it shared exposed that the classifier trains on
  `mteb/banking77@18072d2` while the descriptions were audited against `PolyAI/banking77@796a462`.
  The constants were renamed to `AUDITED_AGAINST_*` (provenance, not a dataset pin); an equivalence
  check over sorted (text, label) pairs is pending.
- **Verifier candidates measured before writing the verifier (#33).** The verifier only chooses
  among the classifier's top-k, so recall@k on low-confidence cases bounds what it can fix. On the
  Banking77 test split, 12.5 % of examples fall below τ and the classifier gets 57.2 % of them right.
  On the ones it gets wrong, the correct label is in the top-3 65.9 % of the time and in the top-5
  82.3 %. Decision: k = 5, and "none of these" maps to `unknown`. Cost is not a factor: about
  $0.16 per 1,000 conversations. These are single utterances, so the numbers are optimistic for
  conversations and will be re-measured on the golden set.
- **Process:** large PRs are split into stacks by an agent and reviewed one PR per reviewer
  session; stacks are merged bottom-up with squash, `rebase --onto` with `--update-refs`, and
  `push --force-with-lease --atomic`, driven by a script that stops before any push if the content
  changed.
- Next: wire `build_pipeline(settings)` into the API lifespan (end-to-end service), then the LLM
  intent verifier behind an evaluation gate.
