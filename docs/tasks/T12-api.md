# T12 — FastAPI service (part 1: API without persistence)

**Lane:** H · **Depends on:** T01, T05 (T10 is consumed through a Protocol, see below) ·
**Issue:** #17 (hardening items that belong to the API)

## Goal
Serve the pipeline over HTTP: analyze conversations posted by clients and by Twilio, expose
health, readiness and metrics, and enforce the ingest hardening that T01 deferred to the API.
Part 1 has no storage; results are returned to the caller (or only traced and measured for
Twilio callbacks, which ignore response bodies).

## Allowed paths
`src/convintel/api/`, `tests/api/`, `docs/tasks/T12-api.md`, `pyproject.toml`, `uv.lock`

## Requirements
- `convintel.api.main:create_app(...)` builds the FastAPI app; `convintel.api.main:app` is the
  module-level instance used by `uvicorn`.
- The pipeline (T10, not on `main` yet) is consumed through a `Pipeline` Protocol defined in
  `convintel.api` (`async def analyze(conversation) -> ConversationInsights`,
  `async def warmup() -> None`), injected through FastAPI dependencies. No
  `src/convintel/pipeline/` in this task.
- `POST /v1/conversations`: body is a `Conversation`, response is `ConversationInsights`.
  A conversation with an `audio_uri` and no turns is transcribed first (through the
  `Transcriber` contract) after the spool checks below.
- Twilio webhooks wired to the T01 ingest code, signature verification exactly as T01 built it
  (`validate_twilio_webhook`, no bypass flag):
  - `POST /v1/twilio/sms` → `normalize_twilio_sms` → pipeline → empty TwiML.
  - `POST /v1/twilio/recording` → `normalize_twilio_recording` → recording fetcher (Protocol,
    writes into the spool) → spool checks → transcriber → pipeline → `204`.
  - Webhook URL rebuilt from an explicit public base URL, never from forwarded headers;
    repeated form keys and oversized bodies rejected before building `dict[str, str]`.
- `GET /healthz` (liveness, always 200 while the process serves), `GET /readyz` (200 only after
  `warmup()` succeeded, 503 before and during shutdown), `GET /metrics` (Prometheus text from the
  existing default registry).
- Lifespan: calls `convintel.telemetry.setup()` and `pipeline.warmup()`; startup fails if warmup
  fails or no pipeline is configured.
- Request metrics with the existing `convintel_requests_total{route,status}` (never redeclared),
  `route` = route template, never the raw path.
- Spans `api.<route>` with ids, channel and status only.
- Error mapping: `ConvIntelError` subclasses → 5xx with a fixed message per type; unexpected
  exceptions → 500 with a fixed message; request validation errors → 422 without echoing input.
- Issue #17 items:
  - Audio size limit (bytes) and duration limit (container header probe, and `RecordingDuration`
    before fetching) checked before any decoding.
  - Audio paths confined to a spool directory (resolved, symlinks followed, regular files only).
  - Twilio webhooks require `TWILIO_ACCOUNT_SID` (and `TWILIO_AUTH_TOKEN`); without them the
    endpoints fail closed with 503.
- Settings needed by the API are module constants or explicit `create_app` parameters;
  `config.py` is human-owned. The exact `Settings` fields to add go in the PR.

## Acceptance criteria
- [ ] Unit tests with `TestClient` and fakes (pipeline, transcriber, fetcher); no network.
- [ ] Tests prove: valid / invalid Twilio signature, wrong account SID, missing Twilio config,
      repeated form keys, oversize body, spool escape (`..`, absolute path, symlink), oversize and
      overlong audio, readiness before/after warmup, startup failure on warmup failure,
      `convintel_requests_total` increments with the route template, 5xx bodies are fixed strings.
- [ ] A test asserts that synthetic conversation text never appears in responses, log records,
      span attributes or metric output for failing requests.
- [ ] `ruff`, `ruff format --check`, `pyright`, `pytest`, `uv lock --check` pass.
- [ ] PR lists the `Settings` fields requested from the human (name, type, default, validation).

## Out of scope
Persistence / Postgres and idempotency storage (T12 part 2). Auth beyond Twilio signatures.
Deployment (T13). Downloading Twilio recordings (a fetcher implementation; only the Protocol
ships here). Load tests and SLO dashboards (T15). Editing `config.py`, `contracts/`, `ingest/`
or `telemetry/`.
