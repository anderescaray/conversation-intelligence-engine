import base64
import hashlib
import hmac
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from convintel.config import Settings, get_settings
from convintel.contracts import Channel, Speaker
from convintel.errors import ConvIntelError
from convintel.ingest.normalizers import (
    normalize_chat,
    normalize_twilio_recording,
    normalize_twilio_sms,
)
from convintel.ingest.twilio import TwilioWebhook, validate_twilio_webhook

URL = "https://example.test/webhooks/twilio?source=synthetic"
TOKEN = "synthetic-test-token"
NOW = datetime(2026, 1, 1, tzinfo=UTC)
ACCOUNT = "AC" + "0" * 32


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("TWILIO_AUTH_TOKEN", "TWILIO_ACCOUNT_SID"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return Settings.model_validate(
        {"_env_file": None, "twilio_auth_token": SecretStr(TOKEN), "twilio_account_sid": ACCOUNT}
    )


@pytest.fixture
def sms() -> dict[str, str]:
    return {
        "AccountSid": ACCOUNT,
        "MessageSid": "SM" + "1" * 32,
        "From": "+15005550006",
        "To": "+15005550009",
        "Body": "Please check the synthetic order.",
        "NumMedia": "0",
    }


@pytest.fixture
def recording() -> dict[str, str]:
    return {
        "AccountSid": ACCOUNT,
        "CallSid": "CA" + "2" * 32,
        "RecordingSid": "RE" + "3" * 32,
        "RecordingUrl": "https://api.twilio.com/2010-04-01/Accounts/"
        + ACCOUNT
        + "/Recordings/RE"
        + "3" * 32,
        "RecordingStatus": "completed",
        "RecordingDuration": "10",
        "RecordingChannels": "2",
    }


def signed(params: dict[str, str], url: str = URL) -> TwilioWebhook:
    message = url + "".join(key + params[key] for key in sorted(params))
    signature = base64.b64encode(
        hmac.new(TOKEN.encode(), message.encode(), hashlib.sha1).digest()
    ).decode()
    return TwilioWebhook(url=url, params=params, signature=signature)


def test_sms(settings: Settings, sms: dict[str, str]) -> None:
    result = normalize_twilio_sms(signed(sms), settings=settings, started_at=NOW)
    assert result.channel is Channel.SMS
    assert result.id == sms["MessageSid"]
    assert result.started_at == NOW
    assert result.turns[0].speaker is Speaker.CUSTOMER
    assert result.turns[0].text == sms["Body"]
    assert result.turns[0].index == 0
    assert not result.redacted
    assert sms["From"] not in str(result.metadata)
    assert sms["To"] not in str(result.metadata)


def test_recording(settings: Settings, recording: dict[str, str]) -> None:
    result = normalize_twilio_recording(signed(recording), settings=settings, started_at=NOW)
    assert result.id == recording["CallSid"]
    assert result.channel is Channel.VOICE
    assert result.audio_uri == recording["RecordingUrl"]
    assert result.turns == ()
    assert result.metadata["recording_channels"] == "2"
    assert float(result.metadata["audio_duration_s"]) == 10
    assert result.started_at == NOW


@pytest.mark.parametrize("change", ["body", "extra", "url", "signature", "account"])
def test_rejects_tampering(settings: Settings, sms: dict[str, str], change: str) -> None:
    webhook = signed(sms)
    if change == "body":
        webhook = webhook.model_copy(update={"params": {**sms, "Body": "Changed"}})
    elif change == "extra":
        webhook = webhook.model_copy(update={"params": {**sms, "NewField": "Changed"}})
    elif change == "url":
        webhook = webhook.model_copy(update={"url": URL + "&changed=1"})
    elif change == "account":
        webhook = signed({**sms, "AccountSid": "AC" + "f" * 32})
    else:
        webhook = webhook.model_copy(update={"signature": ""})
    with pytest.raises(ConvIntelError):
        normalize_twilio_sms(webhook, settings=settings)


@pytest.mark.parametrize("signature", ["", "not-base64", "non-ascii-ñ"])
def test_invalid_signature(settings: Settings, sms: dict[str, str], signature: str) -> None:
    with pytest.raises(ConvIntelError, match="signature"):
        validate_twilio_webhook(signed(sms).model_copy(update={"signature": signature}), settings)


def test_missing_token(sms: dict[str, str]) -> None:
    settings = Settings.model_validate({"_env_file": None})
    with pytest.raises(ConvIntelError, match="TWILIO_AUTH_TOKEN"):
        validate_twilio_webhook(signed(sms), settings)


def test_settings_ignore_local_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("TWILIO_AUTH_TOKEN=synthetic-local-token\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert Settings.model_validate({"_env_file": None}).twilio_auth_token is None


def test_recording_requires_signature(settings: Settings, recording: dict[str, str]) -> None:
    webhook = signed(recording).model_copy(update={"signature": ""})
    with pytest.raises(ConvIntelError, match="signature"):
        normalize_twilio_recording(webhook, settings=settings)


def test_signed_unknown_fields_are_not_discarded(settings: Settings, sms: dict[str, str]) -> None:
    validate_twilio_webhook(signed({**sms, "FutureTwilioField": "value"}), settings)


def test_json_webhooks_are_not_mistaken_for_form_webhooks(
    settings: Settings, sms: dict[str, str]
) -> None:
    with pytest.raises(ConvIntelError, match="form"):
        validate_twilio_webhook(signed(sms, URL + "&bodySHA256=abc"), settings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("RecordingDuration", "-1"),
        ("RecordingDuration", "nan"),
        ("RecordingChannels", "3"),
        ("RecordingStatus", "in-progress"),
        ("RecordingUrl", "file:///tmp/private.wav"),
        ("RecordingSid", "invalid"),
    ],
)
def test_invalid_recording(
    settings: Settings, recording: dict[str, str], field: str, value: str
) -> None:
    with pytest.raises(ConvIntelError, match="recording"):
        normalize_twilio_recording(signed({**recording, field: value}), settings=settings)


def test_bad_sms_does_not_leak_payload(settings: Settings, sms: dict[str, str]) -> None:
    with pytest.raises(ConvIntelError) as error:
        normalize_twilio_sms(signed({**sms, "MessageSid": "private-input"}), settings=settings)
    assert "private-input" not in str(error.value)
    assert error.value.__cause__ is None


def chat() -> dict[str, object]:
    return {
        "id": "synthetic-chat",
        "started_at": NOW.isoformat(),
        "language": "es",
        "messages": [
            {"speaker": "customer", "text": "Necesito ayuda con el pedido de prueba."},
            {"speaker": "agent", "text": "Reviso el pedido de prueba."},
        ],
        "metadata": {"source": "synthetic"},
    }


def test_chat() -> None:
    result = normalize_chat(json.dumps(chat()))
    assert result.channel is Channel.CHAT
    assert result.started_at == NOW
    assert result.language == "es"
    assert [turn.index for turn in result.turns] == [0, 1]
    assert [turn.speaker for turn in result.turns] == [Speaker.CUSTOMER, Speaker.AGENT]
    assert not result.redacted
    assert result.metadata == {"source": "synthetic"}


@pytest.mark.parametrize(
    "update",
    [
        {"messages": []},
        {"messages": [{"speaker": "intruder", "text": "x"}]},
        {"started_at": "2026-01-01T00:00:00"},
        {"id": ""},
        {"redacted": True},
        {"messages": [{"speaker": "agent", "text": "   "}]},
    ],
)
def test_invalid_chat(update: dict[str, object]) -> None:
    with pytest.raises(ConvIntelError, match="chat"):
        normalize_chat(json.dumps({**chat(), **update}))


def test_malformed_chat() -> None:
    with pytest.raises(ConvIntelError, match="chat"):
        normalize_chat("not json containing private-input")
