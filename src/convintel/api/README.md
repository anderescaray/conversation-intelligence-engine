# convintel.api

FastAPI service (T12 part 1, no persistence). Build the app with `create_app(...)`; the
module-level `convintel.api.main:app` refuses to start until a pipeline is wired in (T10).

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
- `RecordingFetcher`: `async fetch(recording, spool_dir) -> Path`. No implementation ships yet,
  so the recording webhook answers 503 until one is passed to `create_app`. The returned file
  is deleted after transcription (see spool lifecycle), so return a fresh file per call.
- Transcriber: the `Transcriber` contract; defaults to `WhisperTranscriber()`.

## Safety rules enforced here

- **Twilio**: both `TWILIO_AUTH_TOKEN` and a well-formed `TWILIO_ACCOUNT_SID` are required,
  otherwise 503 (fail closed). Signatures are verified by the T01 helper against
  `public_base_url + path` (or the request URL when unset), never forwarded headers. Bodies over
  64 KiB, non-form content types and repeated fields are rejected before verification.
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
template; unknown paths share `route="unmatched"`.
