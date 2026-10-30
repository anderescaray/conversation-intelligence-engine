# T12 part 2 — Design proposal: async webhooks, idempotency, persistence

**Status:** proposal for review (not an ADR; ADRs are numbered and created by the human).
**Branch context:** `lane-h/T12-api` at `6e696f3` (T12 part 1 done, not merged). No code in this
document has been written; nothing here changes `contracts/` or `config.py`.

**Inputs read:** `AGENTS.md` (this branch and `origin/main`), `docs/ARCHITECTURE.md`,
`docs/PLAN.md`, `docs/tasks/T12-api.md`, `PR-DRAFT-T12.md`, `src/convintel/api/`,
`src/convintel/contracts/schemas.py`, `src/convintel/config.py`, `src/convintel/ingest/`,
`infra/docker-compose.yml`, issue #17, ADR-0001…0005.

**Sources** are cited inline as `[n]` (list at the end). Anything not backed by a citation is
marked **(assumption)** and should be verified before the PR that depends on it.

---

## 0. Facts that shape the design

| # | Fact | Source |
|---|---|---|
| F1 | Twilio webhooks: connect timeout 5 s, read timeout 15 s, total 15 s by default. Voice requests have a hard 15 s limit. | [1] |
| F2 | Twilio retries **once by default and only on TCP connect/TLS failure** (`rc=1`, `rp=ct`). Retries on 4xx/5xx/read timeout happen only if we opt in with connection overrides (`#rc=…&rp=…`, up to 5 retries). | [1] |
| F3 | Overrides travel in the URL **fragment**; Twilio leaves the fragment out of the signature. Our `_public_url` never sees the fragment, so signing is unaffected. Overrides work for all product webhooks except Conversations and Frontline. The signature covers the URL + sorted POST params. | [1], [13] |
| F4 | Cloud Run, request-based billing (the default): "CPU is only allocated during request processing". Instance-based billing keeps CPU on outside requests. | [2] |
| F5 | Cloud Run: work after the response is delivered is "background activity". With request-based billing, CPU is "disabled or severely limited", and the guidance is to finish all async work before responding. | [3] |
| F6 | Cloud Run: idle instances, even ones kept by min-instances, "can be shut down at any time". An idle instance never stays more than 15 min after its last request unless min-instances keeps it. | [2] |
| F7 | Cloud Run request timeout: default 300 s, max 3600 s. After a timeout the client gets a 504 but the instance **may keep processing** the aborted request. | [4] |
| F8 | Cloud Tasks HTTP targets: `dispatchDeadline` default 10 min, range **[15 s, 30 min]**. | [5] |
| F9 | Cloud Tasks task names deduplicate: a duplicate name fails with `ALREADY_EXISTS`. A deleted or executed name stays reserved for up to 24 h. Use hashed names, since sequential names add latency. Max task size is 100 KB. | [6] |
| F10 | Cloud Tasks queue retry config: `max_attempts`, `max_retry_duration`, `min_backoff`/`max_backoff`, `max_doublings`. The example queue shows 100 attempts, 0.1 s to 3600 s backoff and 16 doublings. | [7] |
| F11 | Cloud Tasks → Cloud Run uses OIDC tokens. The task's service account needs the Cloud Run Invoker role. A retry-count header is sent to the handler. | [8] |
| F12 | Cloud SQL: each Cloud Run instance can open up to 100 connections. Cap connections per instance with a pool, and cap instances. Connect through a Unix socket at `/cloudsql/INSTANCE`, a language connector, or private IP. | [9] |
| F13 | psycopg 3: `psycopg_pool.AsyncConnectionPool` ships as a separate package (`psycopg[pool]`). Opening an async pool in the constructor is deprecated: use `open=False` and open/close it in the FastAPI lifespan. It supports `min_size`/`max_size`, `max_idle`, `max_lifetime` and a `check` callback. | [10] |
| F14 | Cloud Run worker pools have no URL and **do not autoscale**: the instance count is set by hand. | [11] |
| F15 | Pub/Sub push redelivers on nack or ack-deadline expiry. The ack deadline of a push message cannot be extended per message. | [12] |

Two consequences matter more than the rest:

* **"Twilio retries" is weaker than it sounds (F2).** By default a 500 or a timeout from us is
  **not** retried. If we want Twilio to be our retry mechanism, we must configure overrides (e.g.
  `#rc=3&rp=all`). Recommendation below: do not depend on that. Once Twilio gets a 2xx, retries are
  our job. Duplicates still happen (connect retry, override retries, operator replays, our own task
  retries), so idempotency is still required.
* **The audio limits from part 1 don't fit inside one task (F8).** `MAX_AUDIO_DURATION_S = 3600`
  at the ASR SLO (RTF p95 < 0.5) is up to 1800 s of transcription alone. That equals the
  Cloud Tasks maximum deadline and leaves no room for fetch + pipeline. Decision D3.

---

## 1. Async processing of Twilio webhooks

Part 1 processes webhooks synchronously (`twilio.py`): fetch → transcribe → pipeline, then 204.
A recording of a few minutes already exceeds F1.

### Options

**A. In-process background tasks** (`BackgroundTasks` / `asyncio.create_task` after the 204).

* With request-based billing, CPU is throttled after the response (F4, F5). Transcription would
  crawl, or freeze until the next request reaches the same instance.
* With instance-based billing + min-instances ≥ 1, CPU stays on, but instances can still be shut
  down at any time (F6) and every deploy replaces them. In-flight work is lost with no record and no
  retry. We already returned 204, so Twilio will not retry either (F2).
* No back-pressure: a burst of recordings starts N concurrent transcriptions in one process
  (memory: full PCM per file, see `audio.py`).
* No new infrastructure. Fine for **local development only**.

**B. Postgres-backed job table + polling workers** (`SELECT … FOR UPDATE SKIP LOCKED`).

* The webhook inserts a job row and returns. Workers claim rows in a transaction.
* Durable and transactional with the dedup row (one `INSERT` covers both), so there is no outbox
  problem. Inspectable with SQL. No GCP-specific dependency.
* On Cloud Run, polling workers need CPU outside requests, which means instance-based billing and
  min-instances ≥ 1. That gives up scale to zero, and idle instances can still be reaped (F6).
  Worker pools don't autoscale (F14). The alternatives are a Cloud Scheduler tick hitting a "drain"
  endpoint (latency = tick period, minimum 1 min **(assumption: Cloud Scheduler's minimum
  granularity is 1 minute)**) or a GKE/VM worker, both outside the target.
* Retries, backoff, leases and visibility timeouts must all be built by us.

**C. Managed queue that pushes to Cloud Run: Cloud Tasks** (recommended), or Pub/Sub push.

* The webhook verifies the signature, writes the dedup/job row, creates a **Cloud Task** (HTTP
  target = a private worker endpoint, OIDC auth, F11) and returns 2xx in milliseconds.
* The work runs **inside a request** to the worker (`POST /internal/jobs/{job_id}`), so CPU is
  allocated with request-based billing (F4) and the worker **scales to zero** when idle. Cloud
  Run's normal autoscaling follows the queue's dispatch rate.
* Retries with exponential backoff, `max_attempts` and rate/concurrency limits are queue config
  (F10). `max_concurrent_dispatches` is the back-pressure knob that protects ASR memory and LLM
  rate limits.
* Task names give a second dedup layer (F9). It is not the guarantee: the window is ≤ 24 h and it
  does not cover our DB.
* Limit: 30 min per attempt (F8) → D3.
* Pub/Sub push is the alternative. It has dead-letter topics, but its ack deadline is short for
  ASR **(assumption: push ack deadline max 600 s)** and redelivery is driven by the deadline
  (F15). Cloud Tasks maps better to "one job, one HTTP call, long deadline, named for dedup".
  **(assumption: Cloud Tasks has no built-in dead-letter queue; the DB status `dead` plays that
  role, see §6.)**

### Topology on Cloud Run (recommendation)

Two Cloud Run **services from the same image**:

| Service | Ingress / auth | Routes | Sizing |
|---|---|---|---|
| `convintel-api` | public (Twilio must reach it), signature-checked | `/v1/conversations`, `/v1/twilio/*`, probes | request-based billing, scale to zero, high concurrency, no ASR model needed for webhooks |
| `convintel-worker` | **private** (IAM: invoker = Cloud Tasks service account only) | `/internal/jobs/{id}`, `/internal/reconcile`, probes | request-based billing, scale to zero, **concurrency 1–2** (ASR memory), timeout ≥ dispatch deadline, CPU/GPU sized for whisper |

The worker must be a separate service: Cloud Run IAM auth is per service. The public webhook
service cannot also demand IAM on `/internal/*` without verifying OIDC tokens in-app, which
would need another dependency and is easy to get wrong. A separate service also lets the ASR
container get different memory/CPU/GPU and a long timeout without affecting the webhook path.
`create_app(role=...)` mounts the right routers (Settings field `api_role`).
`POST /v1/conversations` (synchronous client API) is unchanged in behaviour. It gains persistence (§3).

Locally (no GCP), an `InlineEnqueuer` calls the same job handler in-process with
`asyncio.create_task`. That is option A, limited to dev and tests, behind
`job_queue_backend="inline"`.

### Flow

```
Twilio ──POST──► api: verify signature (403) ─► normalize (400) ─► INSERT ingest_job
                     ON CONFLICT (idempotency_key) DO NOTHING ─► create Cloud Task
                     name = sha256(idempotency_key) ─► 2xx (empty TwiML / 204)

Cloud Tasks ──POST /internal/jobs/{id} (OIDC)──► worker: claim job (lease) ─► fetch from Twilio
   ─► spool checks ─► transcribe ─► delete audio ─► pipeline.analyze ─► INSERT insights +
   mark job succeeded (one transaction) ─► 200 (ack)   | retryable error ─► 503 (Cloud Tasks retries)
                                                       | permanent error ─► mark failed ─► 200 (ack)
```

### What the job carries: identifiers only

The job row and the task body hold **SIDs, not content**. The worker fetches the content from
Twilio:

* Recording: `RecordingSid` (+ `CallSid`, declared duration, channels). The worker fetches through
  the `RecordingFetcher` Protocol from part 1. Audio exists only during the attempt, which is
  already the part 1 lifecycle (`fetched_recording`).
* SMS: the body is raw PII and would otherwise have to sit in our queue or DB until processed.
  Options: (i) store it in the task payload (raw PII held by Cloud Tasks until the task finishes);
  (ii) process SMS synchronously in the webhook (text-only pipeline, SLO p95 < 4 s, but LLM retries
  can reach 15 s); (iii) **re-fetch the message by `MessageSid` from the Twilio API in the worker**,
  symmetric with recordings. **Recommendation: (iii)**, with (ii) as a fallback if the human prefers
  no extra Twilio call. (iii) needs a `MessageFetcher` from Lane A (request for other lanes). The
  re-fetched body is the authoritative one. **(assumption: the Messages API returns `Body` for
  inbound messages while the message is retained in the Twilio account.)** → D2.

Trace context: the API injects `traceparent` into the task headers. The worker starts
`conversation.process` with a **span link** to the webhook span: the ack and the processing are
separate traces, but they stay navigable in Phoenix.

### Recommendation

**C with Cloud Tasks + a private worker service**, the Postgres job table as the source of truth
for state (§2, §6), and `InlineEnqueuer` for local dev. Reject A in production. Keep B as the
fallback if the human rules out GCP-specific services (B costs scale to zero).

---

## 2. Idempotency

### Key

`idempotency_key = "twilio:sms:{MessageSid}"` or `"twilio:recording:{RecordingSid}"`.
The key is the **RecordingSid**, not the CallSid (`Conversation.id`). One call can produce more
than one recording **(assumption: e.g. `<Record>` used twice or recording restarted)**, and each is
a separate delivery. Insights keep `conversation_id = CallSid` (indexed, not unique).

For `POST /v1/conversations`: optional `Idempotency-Key` header, or nothing (the client already
gets the result synchronously). → D5.

### Where the guarantee lives

| Option | Correct under concurrency? | Notes |
|---|---|---|
| **App check** (`SELECT` then `INSERT`) | **No.** Two concurrent deliveries both see "absent". | Simple, but needs a lock or a serializable transaction to be correct, and then the constraint is simpler. |
| **Unique constraint** + `INSERT … ON CONFLICT (idempotency_key) DO NOTHING RETURNING id` | **Yes.** Postgres serializes it; exactly one inserter gets a row back. | One statement. The loser does a `SELECT` to read the existing job's status. |
| **Cloud Tasks task name only** (F9) | Partial | The window is ≤ 24 h and it says nothing about the DB. Defence in depth only. |

**Recommendation:** the unique constraint is the guarantee. The task name
`sha256(idempotency_key)` (hashed, F9) is the second layer and makes "re-enqueue" idempotent for
the reconciler (§6). Signature verification comes **before** the insert, so an unsigned caller
cannot pre-claim keys.

Also store `payload_sha256` (hash of the canonical, sorted signed params, never the params). If a
duplicate key arrives with a different hash, keep the first, increment
`convintel_ingest_duplicates_total{source, outcome="conflict"}` and log ids only.

### What the second delivery returns

Always the **same success response as the first**: empty TwiML for SMS, `204` for recordings.
This holds whether the first job is queued, running, succeeded, failed or dead.

* A 409 is a 4xx. With `rp=4xx|all` (F2) it would trigger Twilio retries, and it shows up as an
  error in the Twilio Debugger for something that is not an error.
* A retry from Twilio means "Twilio is unsure we got it". We did, so we say so.
* If the first attempt **failed before** the row was committed (DB down), there is no row and the
  retry is processed normally. That is the point of letting Twilio retry 5xx (D4).
* A duplicate never re-enqueues by itself. Re-processing a `failed`/`dead` job is an explicit
  operator action (§6). Otherwise a poison message would be retried forever through the front door.

Metric: `convintel_ingest_duplicates_total{source, outcome="duplicate"|"conflict"}`.

---

## 3. Persistence of `ConversationInsights`

### Schema: columns vs JSONB

| Option | Pros | Cons |
|---|---|---|
| **All JSONB** (one `insights jsonb`) | Zero mapping; contract changes need no migration. | Dashboards query JSON paths; expression indexes everywhere; types not enforced by the DB. |
| **All columns** (normalized: operator results in child tables) | Typed, indexable, SQL-friendly. | Every contract change is a migration; `top_k`, `action_items` and `customer_trajectory` are awkward; lots of mapping code. |
| **Hybrid** (recommended) | Columns for what dashboards filter/aggregate on; the full model as JSONB for fidelity and replay. | Two representations; the columns are derived from the JSONB in one function, so there is one writer. |

Proposed tables (names indicative; DDL lands in the migration PR):

**`ingest_job`**: one row per accepted delivery (and per `POST /v1/conversations` if D5 adds it)

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` PK | task body carries only this |
| `idempotency_key` | `text` **UNIQUE NOT NULL** | §2 |
| `source` | `text` CHECK in (`twilio_sms`, `twilio_recording`, `api`) | |
| `channel` | `text` | `Channel` value |
| `conversation_id` | `text` NOT NULL | MessageSid / CallSid |
| `external_ref` | `jsonb` | **identifiers only**: RecordingSid, declared duration, channels. Never a body, never a URL with credentials |
| `payload_sha256` | `bytea` | §2 |
| `status` | `text` CHECK in (`queued`, `running`, `succeeded`, `failed`, `dead`) | |
| `attempts` | `int` NOT NULL default 0 | incremented on claim |
| `lease_expires_at` | `timestamptz` NULL | §6 |
| `last_error_type` | `text` NULL | `type(exc).__name__` only, never a message |
| `created_at`, `updated_at`, `finished_at` | `timestamptz` | |

Indexes: unique on `idempotency_key`; `(status, updated_at)` for the reconciler and dead-letter
view; `(conversation_id)` for erasure.

**`conversation_insights`**: one row per successful job

| Column | Type | Dashboard use |
|---|---|---|
| `id` | `uuid` PK | |
| `job_id` | `uuid` **UNIQUE** FK → `ingest_job` | makes "write result" idempotent |
| `conversation_id` | `text` NOT NULL | lookup, erasure |
| `channel` | `text` | filter |
| `started_at`, `processed_at` | `timestamptz` | time axis |
| `pipeline_version` | `text` | canary comparison (T14) |
| `intent_label`, `intent_source`, `intent_confidence` | `text`, `text`, `real` NULL | top intents, verifier share |
| `sentiment_overall` | `text` NULL | trend |
| `resolved` | `boolean` NULL | resolution rate |
| `escalation_risk` | `real` NULL | escalation queue |
| `pii_entity_count` | `int` | |
| `total_latency_ms`, `total_cost_usd` | `double precision` | cost dashboards |
| `trace_id` | `text` NULL | link to Phoenix |
| `errors` | `text[]` | `"<operator>:<ErrorType>"` strings (T10 contract) |
| `insights` | `jsonb` NOT NULL | full `ConversationInsights.model_dump(mode="json")` |

Indexes for supervisor dashboards (T15/T16), all time-bounded:
`(processed_at)`, `(intent_label, processed_at)`, `(sentiment_overall, processed_at)`,
`(channel, processed_at)`, `(pipeline_version, processed_at)`, and a partial index
`(processed_at) WHERE escalation_risk >= 0.7 OR resolved IS FALSE` for the "needs attention"
view. No GIN on `insights` until a query needs it. Revisit with real dashboard queries in T15
(`EXPLAIN` on synthetic volume), not before.

Not stored: the summary as a column (it is inside `insights` and has no dashboard filter), turn
text (§4).

### Driver

| Option | Pros | Cons |
|---|---|---|
| **psycopg 3** (async) + `psycopg_pool` | One driver for both async (app) and sync (migrations, scripts); libpq semantics; pool built for FastAPI lifespan (F13). | Slightly slower than asyncpg in microbenchmarks **(assumption)**. Irrelevant next to ASR/LLM latency. |
| **asyncpg** | Fast; mature. | Async only, so migrations need a second driver. Its prepared-statement cache conflicts with transaction-mode poolers **(assumption: relevant if Cloud SQL managed pooling or PgBouncer in transaction mode is used)**. |
| **SQLAlchemy 2 (async) on either** | Typed query building; table metadata shared with Alembic. | Bigger dependency and abstraction for ~10 queries. |

**Recommendation:** **psycopg 3 + `psycopg_pool`** with hand-written SQL in one small store
module (~10 statements, all reviewed and tested against real Postgres). **Alembic** for
migrations, using its `op.execute`/`op.create_table` API with the sync psycopg driver. Alembic
brings SQLAlchemy as a dependency, but the app does not import it at runtime. Alembic 1.20 and
SQLAlchemy 2.1 are already in `uv.lock` transitively (through `mlflow` in the `ml` group), so
the base install gains declarations, not unknown packages. New base deps: `psycopg[binary,pool]`,
`alembic`. Justified in the PR. → D7.

Note: `Settings.database_url` defaults to `postgresql://…` (a libpq URL). psycopg takes it as is.
Alembic needs `postgresql+psycopg://…`, which `env.py` derives so there is still one setting.
The URL carries the password, so it should be `SecretStr` (human-owned change, D8).

### Migrations

* `migrations/` at the repo root (Alembic), linear revisions, written by hand (no autogenerate:
  no ORM models).
* **Never run at app startup.** With N Cloud Run instances starting at once, startup migrations
  race, and a failed migration crash-loops the service. Run them as a **Cloud Run job** (same
  image, `alembic upgrade head`) in the CD pipeline before the traffic shift. That belongs to T13
  and is listed as a follow-up. Locally: `uv run alembic upgrade head`.
* Expand/contract for every change, because canaries (T14) run two `pipeline_version`s against
  one database.
* The app checks at startup that the DB revision equals the code's expected head and **fails
  readiness** otherwise (a deploy before the migration is loud, not silent corruption).

### Connection pooling on Cloud Run

* One `AsyncConnectionPool(open=False)` per process, opened/closed in the lifespan (F13), stored
  in `ApiState` and exposed through a dependency, like the other seams.
* Sizing rule: `max_instances × db_pool_max_size ≤ Cloud SQL max_connections − headroom`
  (migrations, admin, MLflow on the same instance per `docker-compose.yml`). F12 caps each Cloud
  Run instance at 100 connections and recommends capping per instance with a pool. Suggested start:
  api `max_size=5`, worker `max_size=2` (concurrency 1–2, and a job holds a connection only for
  its short transactions, **never across ASR/LLM calls**).
* `min_size=0` or 1 so idle instances don't pin connections. `max_idle` ≈ a few minutes;
  `max_lifetime` < any server-side idle cutoff; `check=AsyncConnectionPool.check_connection` to
  survive instances that were CPU-throttled while idle (F4) **(assumption: connections can go
  stale across throttled periods)**.
* Connection path: Cloud SQL Unix socket `/cloudsql/…` (F12, no extra dependency, encrypted).
  The Python connector would be another dependency. Decided in T13.
* `statement_timeout` per connection (e.g. 5 s) so a stuck query cannot hold a webhook past 15 s.
* `/readyz` stays 200 when the DB is down (do not take the whole API out for a DB blip).
  Webhooks answer 503 and Twilio retries according to D4. `/healthz` is unchanged.

---

## 4. Privacy

### What we store

| Store | Content | Raw PII? |
|---|---|---|
| `ingest_job` | SIDs, declared duration, hashes, status, error type names | No content. SIDs are pseudonymous identifiers (personal data under GDPR when linkable through Twilio **(assumption, legal review)**). |
| `conversation_insights` | Labels, scores, costs, summary/action items **generated from redacted text** | No raw PII by construction, but Presidio can miss entities, so the LLM summary can echo one. Treat the table as personal data for retention and erasure. |
| Cloud Tasks | `job_id` only | No |
| Audio spool | Raw audio during one attempt | **Yes.** Deleted in `finally` (part 1), TTL sweep as safety net |
| Phoenix (traces) | Ids in our spans; OpenInference LLM spans carry prompts = **redacted** text | Same caveat as summaries; Phoenix retention is part of the erasure story (Lane E) |
| Twilio account | Original recordings and message bodies | **Yes**, outside our DB but under our account (D9) |

Not stored by design: phone numbers (`From`/`To` are not in the normalizer output and stay out
of `external_ref`), raw turn text, recording URLs.

### Store turn text?

| Option | Pros | Cons |
|---|---|---|
| **Don't store turns** (recommended for part 2) | Smallest PII surface; nothing to re-redact if Presidio improves. | T16 "transcript search" only has summaries; no re-analysis without re-fetching from Twilio. |
| Store **redacted** turns (`conversation_turn` table) | Enables T16 search and offline re-runs. | **Needs a contract change:** `ConversationInsights` has no redacted conversation (only `pii_entity_count`; the redacted `Conversation` lives in `PIIOutput`, inside the pipeline). Either `Pipeline.analyze` returns it too (our Protocol, but T10 must agree) or `ConversationInsights` grows a field (frozen contract → human). Residual-PII risk at the largest volume. |
| Store redacted turns with a **shorter retention** than insights | Middle ground for T16. | Two retention clocks. |

Recommendation: part 2 stores **no turn text**. T16 decides whether it needs redacted turns
and asks for the contract change then. → D6.

### Retention and deletion (GDPR)

* **Retention** (Settings, enforced by a sweep): `ingest_job` rows ~30 days (dedup only needs to
  outlive Twilio's retry horizon, which is minutes), `conversation_insights` N days (product
  decision, D10), spool 24 h (existing).
* **Sweep:** `/internal/maintenance/sweep` on the worker, triggered by Cloud Scheduler
  (daily), deleting in batches. It also runs the periodic spool sweep that part 1 deferred to T13.
* **Erasure (right to be forgotten):** a single function `erase_conversation(conversation_id)`
  deletes insights + jobs in one transaction and any spool file for that job. Spool files are named
  `{job_id}.{ext}` by the fetcher, so they can be targeted without listing.
  * Gap: we don't store phone numbers, so a request "delete everything about +34…" cannot be
    resolved from our DB. It needs a Twilio lookup (messages/calls by number → SIDs). This is a
    feature for privacy (no phone index in our DB), but the runbook must say so. → D11.
  * Backups: Cloud SQL backups/PITR keep deleted rows until they age out **(assumption: retention
    window configured in T13)**. Document it as the erasure SLA.
  * Phoenix and Twilio-held data: requests for Lanes E / A (D9).
* **No erasure HTTP endpoint in part 2** (auth beyond Twilio signatures is out of scope). An
  admin script, run as a Cloud Run job. → D11.

---

## 5. Testing without network

Seams, consistent with part 1 (Protocols injected through `create_app` / `ApiState`):

* `JobStore`: `create_or_get(key, …) -> (Job, created: bool)`, `claim(job_id, lease_s)`,
  `complete(job_id, insights)`, `fail(job_id, error_type, permanent)`, `requeue_stuck(older_than)`,
  `sweep(...)`, `erase(conversation_id)`.
* `Enqueuer`: `enqueue(job) -> None` (idempotent by task name).
* `MessageFetcher` (Lane A) next to the existing `RecordingFetcher`.

| Layer | How | Proves |
|---|---|---|
| **Unit** (default `pytest -q`, sockets blocked by the existing autouse fixture) | `InMemoryJobStore` (dict + `asyncio.Lock`, same uniqueness semantics), `FakeEnqueuer` (records tasks, can fail on demand), existing fakes | Webhook acks before processing; duplicate → same response, no second enqueue; worker status transitions; retryable → 503, permanent → 200 + `failed`; attempts cap → `dead`; spool file gone after every outcome; enqueue failure → job stays `queued` and the reconciler re-enqueues; no conversation text in responses/logs/spans/metrics (extend part 1's leak test to the worker) |
| **Contract tests** | One parametrized suite run against `InMemoryJobStore` (unit) **and** `PostgresJobStore` (integration) | The fake cannot drift from the real store |
| **Integration** (`@pytest.mark.integration`, Postgres from `infra/docker-compose.yml`) | Session fixture creates a throwaway database (`convintel_test_<random>`), runs `alembic upgrade head`, drops it at the end; never touches the dev `convintel` DB | Real `ON CONFLICT` under **concurrent** inserts (N tasks → 1 row); lease expiry and re-claim; `job_id` unique on insights; migrations `upgrade → downgrade → upgrade`; revision check at startup; pool open/close in the lifespan |
| **Cloud Tasks client** | Unit: the request built for `create_task` (name hash, OIDC config, deadline, `traceparent` header) against a fake transport; no emulator **(assumption: no official Cloud Tasks emulator)** | Correct task shape, `ALREADY_EXISTS` treated as success |

Notes:
* Per `origin/main` AGENTS.md, the Docker stack is started **from the main checkout, never from
  a worktree**. Integration tests only read `DATABASE_URL` via `Settings(_env_file=None, …)`.
* The no-network fixture needs a localhost exemption for integration tests only.
* CI runs `-m 'not integration'`. A Postgres service container job in GitHub Actions would cover
  the contract and concurrency tests on every PR → D12 (CI is not in this lane).

---

## 6. Failure modes

### Classification (worker)

| Failure | Class | Worker response | Job status |
|---|---|---|---|
| `AudioRejectedError` (`too_large`, `too_long`, `unreadable`, `outside_spool`) | permanent | 200 | `failed` |
| Twilio fetch 404 (recording/message deleted) | permanent | 200 | `failed` |
| Twilio fetch 5xx/timeout, 429 | retryable | 503 | `queued` (attempts+1) |
| `TranscriptionError` | retryable up to the cap **(assumption: corrupt input is caught earlier by the header probe)** | 503 | → `dead` at cap |
| `LLMError` (after the client's own retries) | retryable | 503 | → `dead` at cap |
| Pipeline returns insights with `errors` (T10 degrades per operator) | **success** | 200 | `succeeded`, `errors` stored and metric emitted. No retry: partial results are a product outcome, and retrying burns LLM cost |
| DB unavailable | retryable | 503 | unchanged |
| Unexpected exception (bug) | retryable up to the cap | 500 | → `dead` at cap |

Cloud Tasks retries any non-2xx (F10, F11). We keep **our own** `attempts` counter in the DB so
the cap holds even if queue config drifts: `job_max_attempts` (e.g. 5) < queue `max_attempts`.
When the DB counter reaches the cap, the worker marks `dead` and returns 200 to stop the queue.

### Scenarios

1. **Pipeline fails after the webhook was acknowledged.** Twilio is out of the loop (F2). The job
   stays `queued` and Cloud Tasks retries with backoff. The audio was deleted in the failed
   attempt, and the retry refetches it from Twilio (part 1 design, PR-DRAFT §2). After the cap:
   `dead`, metric `convintel_jobs_total{outcome="dead"}`, alert (T15).
2. **Enqueue fails after the job row was committed** (outbox gap). The webhook returns 2xx anyway:
   the row is durable. A reconciler (`/internal/reconcile`, Cloud Scheduler every few minutes)
   re-enqueues `queued` jobs older than `job_stuck_after_s`. Task-name dedup (F9) makes that a
   no-op for jobs whose task exists. **Alternative:** return 503 so Twilio retries (only with
   overrides, D4). Rejected as the primary path, because the reconciler works without Twilio's help.
3. **Duplicate jobs / concurrent execution.** Possible when the Cloud Tasks deadline expires while
   the instance keeps working (F7), or with a re-enqueue. `claim` is
   `UPDATE … SET status='running', attempts=attempts+1, lease_expires_at=now()+lease
   WHERE id=$1 AND (status='queued' OR (status='running' AND lease_expires_at<now())) RETURNING …`.
   A non-claimed attempt returns **503**, so the queue retries later; it cannot know if the other
   attempt will succeed. A terminal job returns **200**. Lease > dispatch deadline. The result write
   is `INSERT … ON CONFLICT (job_id) DO NOTHING` + status update in one transaction, so a late
   duplicate cannot write a second result.
4. **Poison message.** Bounded by the DB attempt cap → `dead`. Never re-enqueued by duplicates
   (§2) or by the reconciler (it only touches `queued`). Replay = explicit admin script
   (`status='queued', attempts=0` + enqueue) after a fix. `dead` jobs are the dead-letter queue,
   queryable by `last_error_type`.
5. **DB down at webhook time.** 503 before anything is stored. Twilio retries only if overrides
   include 5xx (D4). Otherwise the delivery is lost, visible in the Twilio Debugger and in
   `convintel_requests_total{status="503"}`.
6. **Worker instance killed mid-job** (scale-in, deploy). The request fails, Cloud Tasks retries,
   and the lease expires, so the next attempt claims. The spool file is left behind, and the TTL
   sweep removes it.
7. **Long audio exceeds 30 min** (F8). The attempt times out, retries repeat the same timeout, and
   the job ends `dead`, burning `max_attempts × 30 min` of compute. Prevent it at the webhook:
   reject on declared `RecordingDuration` above an async-path limit (D3).

### Metrics (new; names follow `docs/ARCHITECTURE.md#telemetry`)

`convintel_jobs_total{source, outcome}` (succeeded/failed/dead/retried),
`convintel_job_queue_delay_seconds{source}` (created → first claim),
`convintel_job_attempts{source}` (histogram at terminal state),
`convintel_ingest_duplicates_total{source, outcome}`. Spans: `api.twilio.*` (ack path),
`worker.job` (with `job.id`, `conversation.id`, `job.attempt`, `error.type` only), a link to the
webhook span, and `conversation.process` from T10 under it. Declaring the metrics in
`convintel.telemetry` is a request to Lane E (no redeclaration in `api/`).

---

## 7. Proposed PR split

Each PR is green on its own, ≲ 400 lines where possible, stacked on T12 part 1.

| # | Branch | Contents | Deps |
|---|---|---|---|
| P1 | `lane-h/T12e-jobstore-fakes` | `JobStore`/`Enqueuer` Protocols, `Job`/`JobStatus` models, `InMemoryJobStore`, `FakeEnqueuer`, contract test suite (fake only), brief `docs/tasks/T12-part2.md` | none |
| P2 | `lane-h/T12f-postgres-store` | `alembic` setup + revision 0001 (`ingest_job`, `conversation_insights`, indexes), `PostgresJobStore`, pool in lifespan + revision check, contract suite against Postgres (integration), concurrency test | `psycopg[binary,pool]`, `alembic` |
| P3 | `lane-h/T12g-async-webhooks` | Webhooks → dedup + enqueue + ack; `/internal/jobs/{id}` worker route with claim/lease/classification; `InlineEnqueuer`; `api_role` router mounting; leak test extended | none |
| P4 | `lane-h/T12h-cloud-tasks` | `CloudTasksEnqueuer` (hashed names, OIDC, deadline, `traceparent`), `ALREADY_EXISTS` handling, `/internal/reconcile` | `google-cloud-tasks` (D1) |
| P5 | `lane-h/T12i-persist-sync` | `POST /v1/conversations` persists insights (+ `Idempotency-Key` if D5 says so) | none |
| P6 | `lane-h/T12j-retention` | Retention sweep endpoint, periodic spool sweep, `erase_conversation` + admin script, README/runbook section | none |

Requests for other lanes (to file in the PRs): Lane A `MessageFetcher` + `RecordingFetcher`
implementations (Twilio REST, by SID, byte-capped, writing `{job_id}.{ext}`); Lane E metric
declarations + Phoenix retention; T10 agreement on the result shape if D6 changes; T13 Cloud
Run services, Cloud Tasks queue, Cloud Scheduler jobs, migration job, Cloud SQL sizing.

Path ownership: persistence code does not fit the current repo map (`api/` is T12's lane). The
proposal is `src/convintel/storage/` + `migrations/` added to T12's allowed paths. Human decision
(D13).

---

## 8. `Settings` fields needed (config.py is human-owned: list only)

In addition to those requested in `PR-DRAFT-T12.md`.

| Field | Type | Default | Validation / notes |
|---|---|---|---|
| `database_url` (existing) | `SecretStr` (was `str`) | unchanged | carries a password (D8) |
| `db_pool_min_size` | `int` | `0` | `ge=0` |
| `db_pool_max_size` | `int` | `5` | `ge=1`, `≥ min`; see sizing rule §3 |
| `db_pool_timeout_s` | `float` | `5.0` | `gt=0`; must keep a webhook well under 15 s |
| `db_statement_timeout_ms` | `int` | `5000` | `gt=0` |
| `api_role` | `Literal["api", "worker", "all"]` | `"all"` | which routers `create_app` mounts; `all` for local dev |
| `job_queue_backend` | `Literal["inline", "cloud_tasks"]` | `"inline"` | `cloud_tasks` requires the three fields below (model validator) |
| `cloud_tasks_queue` | `str \| None` | `None` | full path `projects/…/locations/…/queues/…` (pattern) |
| `cloud_tasks_worker_url` | `HttpUrl \| None` | `None` | https; also the OIDC audience |
| `cloud_tasks_service_account` | `str \| None` | `None` | email pattern; same project as the queue (F11) |
| `job_dispatch_deadline_s` | `float` | `1800.0` | `ge=15, le=1800` (F8) |
| `job_lease_s` | `float` | `1900.0` | `gt=job_dispatch_deadline_s` (validator) |
| `job_max_attempts` | `int` | `5` | `ge=1` |
| `job_stuck_after_s` | `float` | `600.0` | `gt=0`; reconciler threshold |
| `async_max_audio_duration_s` | `float` | `1200.0` | `gt=0`, `≤ audio_max_duration_s`; D3 |
| `jobs_retention_days` | `int` | `30` | `ge=1` |
| `insights_retention_days` | `int` | `90` | `ge=1`; D10 |

Each gets a `.env.example` entry (the test enforces it).

---

## 9. Decisions for the human

| # | Decision | Recommendation |
|---|---|---|
| D1 | Queue: Cloud Tasks (C) vs Postgres job table + polling (B) vs Pub/Sub. Adds `google-cloud-tasks` and ties async processing to GCP. | Cloud Tasks + Postgres state |
| D2 | SMS content transport: re-fetch by SID (iii) vs raw body in the task (i) vs synchronous SMS (ii) | (iii) |
| D3 | Long recordings vs 30 min task deadline: lower the async limit (e.g. 20 min), GPU worker, or Cloud Run jobs for long audio | Lower limit now (`async_max_audio_duration_s`), GPU decision in T13 |
| D4 | Configure Twilio connection overrides (e.g. `#rc=3&rp=all`)? Affects scenario 5 | Yes: cheap, the fragment is not signed (F3). Twilio console/T13 |
| D5 | `Idempotency-Key` on `POST /v1/conversations`? | Not now; persist each call |
| D6 | Store redacted turn text (needs a pipeline/contract change)? | No in part 2; revisit with T16 |
| D7 | Driver/migrations: psycopg 3 + raw SQL + Alembic vs SQLAlchemy Core vs asyncpg | psycopg 3 + Alembic |
| D8 | Make `database_url` a `SecretStr` | Yes |
| D9 | Delete recordings/messages at Twilio after successful processing (data minimisation), and Phoenix trace retention | Yes for recordings, after `succeeded` (Lane A); Phoenix retention ≤ insights retention |
| D10 | Insights retention period | Product/legal call; 90 days as a placeholder |
| D11 | Erasure: admin script only, or an authenticated endpoint; documented lookup by phone via Twilio | Script now |
| D12 | Run Postgres integration tests in CI (service container) | Yes, separate job |
| D13 | Where persistence code lives (`src/convintel/storage/`, `migrations/`) and who owns it | Add to T12's allowed paths |
| D14 | Separate worker Cloud Run service vs in-app OIDC verification on one service | Separate service |

---

## Sources

1. Twilio, *Webhooks connection overrides*: https://www.twilio.com/docs/usage/webhooks/webhooks-connection-overrides
2. Google Cloud, *Cloud Run billing settings*: https://docs.cloud.google.com/run/docs/configuring/billing-settings
3. Google Cloud, *Cloud Run general development tips*: https://docs.cloud.google.com/run/docs/tips/general
4. Google Cloud, *Cloud Run request timeout*: https://docs.cloud.google.com/run/docs/configuring/request-timeout
5. Google Cloud, *Cloud Tasks REST reference: Task (`dispatchDeadline`)*: https://docs.cloud.google.com/tasks/docs/reference/rest/v2/projects.locations.queues.tasks
6. Google Cloud, *Cloud Tasks `tasks.create`*: https://docs.cloud.google.com/tasks/docs/reference/rest/v2/projects.locations.queues.tasks/create
7. Google Cloud, *Configure Cloud Tasks queues*: https://docs.cloud.google.com/tasks/docs/configuring-queues
8. Google Cloud, *Create HTTP target tasks*: https://docs.cloud.google.com/tasks/docs/creating-http-target-tasks
9. Google Cloud, *Connect from Cloud Run to Cloud SQL for PostgreSQL*: https://docs.cloud.google.com/sql/docs/postgres/connect-run
10. psycopg 3, *Connection pools*: https://www.psycopg.org/psycopg3/docs/advanced/pool.html
11. Google Cloud, *Deploy Cloud Run worker pools*: https://docs.cloud.google.com/run/docs/deploy-worker-pools
12. Google Cloud, *Pub/Sub push subscriptions*: https://docs.cloud.google.com/pubsub/docs/push
13. Twilio, *Webhook security (signature)*: https://www.twilio.com/docs/usage/security
