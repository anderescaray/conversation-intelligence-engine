"""Fakes and helpers shared by the API tests. All content is synthetic."""

import wave
from datetime import UTC, datetime
from pathlib import Path

from convintel.contracts import Channel, Conversation, ConversationInsights, Speaker, Turn

TOKEN = "synthetic-test-token"
ACCOUNT = "AC" + "0" * 32


class FakePipeline:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.warmup_error: Exception | None = None
        self.warmed = False
        self.seen: list[Conversation] = []

    async def warmup(self) -> None:
        if self.warmup_error is not None:
            raise self.warmup_error
        self.warmed = True

    async def analyze(self, conversation: Conversation) -> ConversationInsights:
        self.seen.append(conversation)
        if self.error is not None:
            raise self.error
        return ConversationInsights(conversation_id=conversation.id, pipeline_version="fake-1")


class FakeTranscriber:
    def __init__(self, redacted: bool = False, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.redacted = redacted
        self.error = error

    async def transcribe(self, audio_uri: str, conversation_id: str) -> Conversation:
        self.calls.append((audio_uri, conversation_id))
        if self.error is not None:
            raise self.error
        return Conversation(
            id=conversation_id,
            channel=Channel.VOICE,
            started_at=datetime.now(UTC),
            turns=(Turn(index=0, speaker=Speaker.CUSTOMER, text="synthetic transcript"),),
            redacted=self.redacted,
            audio_uri=audio_uri,
            metadata={"asr_model": "fake"},
        )


def write_wav(path: Path, seconds: float = 0.5, channels: int = 1, rate: int = 8000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\x00\x00" * channels * int(seconds * rate))
    return path
