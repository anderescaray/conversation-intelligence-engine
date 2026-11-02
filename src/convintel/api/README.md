# convintel.api

FastAPI service (T12 part 1, no persistence). In production, `create_production_app()` is the
composition root: it reads `Settings` and builds the real pipeline (T10) when the server starts.
Serve it with uvicorn's `--factory`:

```bash
uv run uvicorn convintel.api.main:create_production_app --factory
```

There is no module-level app: importing `convintel.api` has no side effects. Tests and custom
wiring build the app with `create_app(...)` and inject their own seams:

```python
from convintel.api import create_app

app = create_app(
    pipeline=my_pipeline, fetcher=my_fetcher, public_base_url="https://api.example.com"
)
```

## Endpoints

| Route | Purpose | Responses |
|---|---|---|
| `POST /v1/conversations` | `Conversation` in, `ConversationInsights` out. No turns + `audio_uri` → transcribed first. | 200, 413 body or audio too large, 422 audio rejected / invalid body, 5xx |
| `POST /v1/twilio/sms` | Incoming SMS webhook → pipeline. | 200 empty TwiML, 400, 403, 413, 415, 503 |
| `POST /v1/twilio/recording` | Recording status callback → fetcher → spool → transcriber → pipeline. | 204, 400, 403, 413, 422, 503 |
| `GET /healthz` | Liveness. | 200 |
| `GET /readyz` | 200 only after `pipeline.warmup()` succeeded. | 200 / 503 |
| `GET /metrics` | Prometheus text, default registry (`convintel.telemetry` metrics). | 200 |

## Seams (`convintel.api.protocols`)

- `Pipeline`: `async analyze(conversation) -> ConversationInsights`, `async warmup()`.
  Warmup runs in the lifespan, after `telemetry.setup()`; if it raises, startup fails.
  Each failure is logged as `"<operator>:<ErrorType>"` (members of the pipeline's
  `ExceptionGroup` one by one; `PipelineError` uses its stage, e.g.
  `"warmup_timeout:PipelineError"`; anything else by type name), never with its message.
- `RecordingFetcher`: `async fetch(recording, spool_dir) -> Path`. No implementation ships yet,
  so the recording webhook answers 503 to correctly signed requests until one is passed to
  `create_app`. The returned file is deleted after transcription (see spool lifecycle), so
  return a fresh file per call.
- Transcriber: the `Transcriber` contract; defaults to `WhisperTranscriber()`.

## Safety rules enforced here

- **Twilio**: both `TWILIO_AUTH_TOKEN` and a well-formed `TWILIO_ACCOUNT_SID` are required,
  otherwise 503 (fail closed). Signatures are verified by the T01 helper against
  `public_base_url + path` (or the request URL when unset), never forwarded headers. Bodies over
  64 KiB, non-form content types and repeated fields are rejected before verification.
  The signature is checked before anything else that can fail with 503: the pipeline and the
  recording fetcher are looked up only after it passes, so an unsigned request gets 403 and
  learns nothing about what is configured. The Twilio credentials are the one exception (no
  signature can be verified without them).
- **Audio** (`AudioSpool`): `audio_uri` must resolve (symlinks followed) to a regular file inside
  the spool directory, at most 128 MiB and 3600 s by its container header. Twilio recordings are
  also rejected by `RecordingDuration` before any download. Confinement is checked before
  existence, so rejections do not reveal files outside the spool.
- **Spool lifecycle**: recordings the API fetched (Twilio) are deleted when transcription ends,
  on success or failure (`recordings.fetched_recording`); Twilio keeps the original and retries,
  so a retry refetches. Files clients placed for `POST /v1/conversations` are never deleted by
  request handling. At startup the lifespan runs `sweep_spool`, which removes regular files
  older than `AUDIO_SPOOL_TTL_S` (24 h) anywhere in the spool, client files included: it is the
  retention bound for raw audio. Deletion never follows symlinks, never deletes them and never
  leaves the spool.
- **Spool trust model**: the spool directory must be writable only by the API process (and by
  whatever the operator uses to place client audio, under the same trust). Confinement is
  checked on the resolved path and the file is opened afterwards; a writer able to swap a file
  for a symlink between those two steps could redirect the read. That race is not defended in
  code: the deployment (T13) must keep the spool private to the API.
- **Pipeline input after a Twilio recording**: the fetched file is deleted as soon as
  transcription ends, so the conversation handed to the pipeline has `audio_uri=None`; the
  pipeline must work from `turns` alone. For `POST /v1/conversations` the client's `audio_uri`
  is kept (the API does not delete client files).
- **Body size**: `POST /v1/conversations` bodies over `MAX_CONVERSATION_BYTES` (1 MiB) get 413
  `Request body too large.` before any JSON parsing (declared `Content-Length` or streamed bytes).
- **Redaction cannot be skipped from outside**: every conversation reaches the pipeline with
  `redacted=False`, whatever the client, Twilio payload or transcriber set. The pipeline (T10)
  must still redact unconditionally and not branch on the flag; this API guard is only the
  first line.
- **No conversation text out of band**: error bodies are fixed strings (validation errors return
  only `loc` and `type`), logs carry exception type names only, spans `api.*` carry
  `conversation.id`, `channel` and `error.type`.

## Error mapping

| Exception | Status | Body `detail` |
|---|---|---|
| `ServiceNotConfiguredError` | 503 | Service not configured. |
| `TranscriptionError` | 500 | Transcription failed. |
| `LLMError` | 502 | An upstream model call failed. |
| `OperatorError` | 500 | Conversation analysis failed. |
| other `ConvIntelError` | 500 | The request could not be processed. |
| any other exception | 500 | Internal server error. |

## Metrics

`convintel_requests_total{route, status}` counts every request except probes (`/healthz`,
`/readyz`, `/metrics`), so the 5xx ratio matches the error-rate SLO. `route` is the route
template; unknown paths share `route="unmatched"`. An exception raised after the response
headers were sent (e.g. inside a streaming body) is counted as `status="500"`, whatever status
was already sent, and re-raised so the server aborts the response; only the exception type is
logged by the API.
