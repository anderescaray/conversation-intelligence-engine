import asyncio
import math
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal
from urllib.parse import unquote, urlsplit

import av
import numpy as np
from av.error import FFmpegError
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model
from numpy.typing import NDArray
from opentelemetry.trace import Span
from pydantic import BaseModel, ConfigDict, Field

from convintel.contracts import Channel, Conversation, Speaker, Turn
from convintel.errors import TranscriptionError
from convintel.ingest._tracing import private_span


class WhisperSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_size: Literal["tiny", "base", "small", "medium", "large-v3"] = "small"
    model_path: Path | None = None
    cache_dir: Path | None = None
    device: Literal["cpu", "cuda"] = "cpu"
    cpu_threads: int = Field(default=4, ge=1)
    language: str | None = Field(default=None, min_length=2)
    stereo_speakers: tuple[Speaker, Speaker] = (Speaker.CUSTOMER, Speaker.AGENT)


class WhisperTranscriber:
    def __init__(self, settings: WhisperSettings | None = None) -> None:
        self.settings = settings or WhisperSettings()
        self._model: WhisperModel | None = None
        self._device = self.settings.device
        self._lock = Lock()

    async def transcribe(self, audio_uri: str, conversation_id: str) -> Conversation:
        return await asyncio.to_thread(self._transcribe, audio_uri, conversation_id)

    def _resolve_model(self) -> Path:
        message = (
            "ASR weights are missing or incomplete. Download them explicitly before starting: "
            'uv run python -c "from faster_whisper.utils import download_model; '
            f"download_model('{self.settings.model_size}')\". "
            "See convintel/transcription/README.md for local-path setup."
        )
        try:
            path = self.settings.model_path
            if path is None:
                path = Path(
                    download_model(
                        self.settings.model_size,
                        local_files_only=True,
                        cache_dir=str(self.settings.cache_dir) if self.settings.cache_dir else None,
                        use_auth_token=False,
                    )
                )
            required = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json")
            if not all((path / name).is_file() for name in required) or not any(
                (path / name).is_file() for name in ("vocabulary.json", "vocabulary.txt")
            ):
                raise TranscriptionError(message)
            return path
        except (OSError, ValueError):
            raise TranscriptionError(message) from None

    def _load_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                str(self._resolve_model()),
                device=self._device,
                compute_type="int8" if self._device == "cpu" else "float16",
                cpu_threads=self.settings.cpu_threads,
                local_files_only=True,
            )
        return self._model

    def _recognize(
        self, channels: tuple[NDArray[np.float32], ...], duration: float
    ) -> tuple[list[Turn], str]:
        model = self._load_model()
        speakers = self.settings.stereo_speakers if len(channels) == 2 else (Speaker.UNKNOWN,)
        turns: list[Turn] = []
        language = self.settings.language or "en"
        for audio, speaker in zip(channels, speakers, strict=True):
            segments, info = model.transcribe(
                audio, language=self.settings.language, vad_filter=True
            )
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                if (
                    not all(
                        math.isfinite(value)
                        for value in (segment.start, segment.end, segment.avg_logprob)
                    )
                    or not 0 <= segment.start <= segment.end
                    or segment.start > duration
                ):
                    raise TranscriptionError("ASR returned invalid segment metadata.")
                if not turns:
                    language = info.language
                turns.append(
                    Turn(
                        index=len(turns),
                        speaker=speaker,
                        text=text,
                        start_s=segment.start,
                        end_s=min(segment.end, duration),
                        asr_confidence=math.exp(min(segment.avg_logprob, 0)),
                    )
                )
        turns.sort(key=lambda turn: turn.start_s or 0)
        return [
            turn.model_copy(update={"index": index}) for index, turn in enumerate(turns)
        ], language

    def _transcribe(self, audio_uri: str, conversation_id: str) -> Conversation:
        with self._lock, private_span("asr.transcribe") as span:
            start = perf_counter()
            span.set_attribute("conversation.id", conversation_id)
            span.set_attribute("asr.model", self.settings.model_size)
            try:
                channels, duration = _decode_local(audio_uri)
                span.set_attribute("audio.duration_s", duration)
                turns, language = self._recognize_with_fallback(channels, duration, span)
            except (OSError, ValueError, RuntimeError, FFmpegError):
                raise TranscriptionError(
                    "ASR failed to decode audio or run the local model. Check the audio, "
                    "model files and CPU int8 runtime; see convintel/transcription/README.md."
                ) from None
            rtf = (perf_counter() - start) / duration
            span.set_attribute("asr.realtime_factor", rtf)
            span.set_attribute("asr.device", self._device)
            return Conversation(
                id=conversation_id,
                channel=Channel.VOICE,
                started_at=datetime.now(UTC),
                language=language,
                turns=tuple(turns),
                audio_uri=audio_uri,
                metadata={
                    "asr_model": self.settings.model_size,
                    "asr_device": self._device,
                    "audio_duration_s": str(duration),
                    "asr_realtime_factor": str(rtf),
                },
            )

    def _recognize_with_fallback(
        self, channels: tuple[NDArray[np.float32], ...], duration: float, span: Span
    ) -> tuple[list[Turn], str]:
        try:
            return self._recognize(channels, duration)
        except (RuntimeError, ValueError):
            if self._device != "cuda":
                raise
            self._device = "cpu"
            self._model = None
            span.set_attribute("asr.fallback", "cpu_int8")
            return self._recognize(channels, duration)


def _decode_local(audio_uri: str) -> tuple[tuple[NDArray[np.float32], ...], float]:
    uri = urlsplit(audio_uri)
    if (
        uri.scheme not in {"", "file"}
        or uri.netloc not in {"", "localhost"}
        or uri.query
        or uri.fragment
    ):
        raise TranscriptionError("ASR accepts local audio paths or local file:// URIs only.")
    path = Path(unquote(uri.path) if uri.scheme else audio_uri)
    options = {"format_whitelist": "wav,mp3,flac,ogg", "protocol_whitelist": "file"}
    chunks: list[NDArray[np.float32]] = []
    with path.open("rb") as audio, av.open(audio, mode="r", options=options) as container:
        streams = container.streams.audio
        if not streams:
            raise TranscriptionError("Input has no audio stream.")
        count = streams[0].codec_context.channels
        if count not in {1, 2}:
            raise TranscriptionError("Only mono or dual-channel audio is supported.")
        resampler = av.AudioResampler(
            format="fltp", layout="stereo" if count == 2 else "mono", rate=16000
        )
        for frame in chain(container.decode(audio=0), [None]):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().astype(np.float32, copy=False))
    if not chunks:
        raise TranscriptionError("Audio must contain samples.")
    decoded = np.concatenate(chunks, axis=1)
    channels = tuple(decoded[index] for index in range(count))
    duration = len(channels[0]) / 16000
    if not duration:
        raise TranscriptionError("Audio must contain samples.")
    return channels, duration
