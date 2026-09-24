# T01 — Ingest and transcription

**Lane:** A · **Depends on:** T00

## Goal
Turn raw inputs (Twilio voice/SMS webhooks, local audio files, chat JSON) into `Conversation` objects.

## Allowed paths
`src/convintel/ingest/`, `src/convintel/transcription/`, `tests/ingest/`, `tests/transcription/`, `tests/fixtures/audio/`

## Requirements
- `ingest.normalizers`: functions from Twilio SMS payload, chat JSON (documented schema) and
  recording metadata to `Conversation` (`channel`, `turns`, `metadata`).
- `transcription.WhisperTranscriber` implementing `Transcriber` with faster-whisper
  (model size from settings, default `small`, `compute_type=int8` on CPU).
- Speaker attribution: stereo split by channel (Twilio dual-channel recordings) as the default.
  pyannote diarization is optional and goes behind a flag.
- Record `asr_confidence` per turn (mean segment probability) and emit `convintel_asr_realtime_factor`.
- Span `asr.transcribe` with `audio.duration_s` and `asr.model` attributes.

## Acceptance criteria
- [ ] Unit tests for every normalizer, using fixtures (no network).
- [ ] Integration test transcribes a ~10 s synthetic WAV fixture and returns ≥ 1 turn per speaker.
- [ ] Real-time factor on CPU for the `small` model documented in the PR.
- [ ] No Twilio signature validation bypass: webhook payload validation helper with a test.

## Out of scope
The HTTP endpoint itself (T12). PII redaction (T02).
