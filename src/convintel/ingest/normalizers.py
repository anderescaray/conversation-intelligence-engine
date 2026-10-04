from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    ValidationError,
)

from convintel.config import Settings
from convintel.contracts import Channel, Conversation, Speaker, Turn
from convintel.errors import ConvIntelError
from convintel.ingest._tracing import private_span
from convintel.ingest.twilio import TwilioWebhook, validate_twilio_webhook

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    speaker: Speaker
    text: NonEmpty = Field(repr=False)


class ChatPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: NonEmpty
    started_at: AwareDatetime
    language: NonEmpty = "en"
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict, repr=False)


class _SMS(BaseModel):
    message_sid: str = Field(alias="MessageSid", pattern=r"^SM[0-9a-fA-F]{32}$")
    body: NonEmpty = Field(alias="Body", repr=False)


class _Recording(BaseModel):
    call_sid: str = Field(alias="CallSid", pattern=r"^CA[0-9a-fA-F]{32}$")
    recording_sid: str = Field(alias="RecordingSid", pattern=r"^RE[0-9a-fA-F]{32}$")
    url: HttpUrl = Field(alias="RecordingUrl", repr=False)
    duration: float = Field(alias="RecordingDuration", ge=0, allow_inf_nan=False)
    channels: Literal["1", "2"] = Field(alias="RecordingChannels")
    status: Literal["completed"] = Field(alias="RecordingStatus")


def normalize_chat(payload: str | bytes | ChatPayload) -> Conversation:
    with private_span("ingest.chat.normalize") as span:
        try:
            chat = (
                payload
                if isinstance(payload, ChatPayload)
                else ChatPayload.model_validate_json(payload)
            )
        except ValidationError:
            raise ConvIntelError(
                "Invalid chat payload; see the documented chat JSON schema."
            ) from None
        span.set_attribute("conversation.id", chat.id)
        return Conversation(
            id=chat.id,
            channel=Channel.CHAT,
            started_at=chat.started_at,
            language=chat.language,
            turns=tuple(
                Turn(index=index, speaker=message.speaker, text=message.text)
                for index, message in enumerate(chat.messages)
            ),
            metadata=chat.metadata,
        )


def normalize_twilio_sms(
    webhook: TwilioWebhook, *, settings: Settings | None = None, started_at: datetime | None = None
) -> Conversation:
    with private_span("ingest.twilio.sms.normalize") as span:
        validate_twilio_webhook(webhook, settings)
        try:
            sms = _SMS.model_validate(webhook.params)
        except ValidationError:
            raise ConvIntelError("Invalid Twilio SMS payload.") from None
        span.set_attribute("conversation.id", sms.message_sid)
        return Conversation(
            id=sms.message_sid,
            channel=Channel.SMS,
            started_at=started_at or datetime.now(UTC),
            turns=(Turn(index=0, speaker=Speaker.CUSTOMER, text=sms.body),),
            metadata={"source": "twilio", "message_sid": sms.message_sid},
        )


def normalize_twilio_recording(
    webhook: TwilioWebhook, *, settings: Settings | None = None, started_at: datetime | None = None
) -> Conversation:
    with private_span("ingest.twilio.recording.normalize") as span:
        validate_twilio_webhook(webhook, settings)
        try:
            recording = _Recording.model_validate(webhook.params)
            if recording.url.scheme != "https" or recording.url.username or recording.url.password:
                raise ValueError
        except (ValidationError, ValueError):
            raise ConvIntelError("Invalid Twilio recording metadata.") from None
        span.set_attribute("conversation.id", recording.call_sid)
        return Conversation(
            id=recording.call_sid,
            channel=Channel.VOICE,
            started_at=started_at or datetime.now(UTC),
            turns=(),
            audio_uri=str(recording.url),
            metadata={
                "source": "twilio",
                "recording_sid": recording.recording_sid,
                "recording_channels": recording.channels,
                "audio_duration_s": str(recording.duration),
            },
        )
