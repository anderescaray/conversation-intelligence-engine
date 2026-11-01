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
| `GET /healthz` | Liveness. | 200 |
| `GET /readyz` | 200 only after `pipeline.warmup()` succeeded. | 200 / 503 |
| `GET /metrics` | Prometheus text, default registry (`convintel.telemetry` metrics). | 200 |

## Seams (`convintel.api.protocols`)

- `Pipeline`: `async analyze(conversation) -> ConversationInsights`, `async warmup()`.
  Warmup runs in the lifespan, after `telemetry.setup()`; if it raises, startup fails.
- Transcriber: the `Transcriber` contract; defaults to `WhisperTranscriber()`.

## Safety rules enforced here

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
