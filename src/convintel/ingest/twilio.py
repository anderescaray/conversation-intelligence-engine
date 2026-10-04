from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field
from twilio.request_validator import RequestValidator

from convintel.config import Settings, get_settings
from convintel.errors import ConvIntelError
from convintel.ingest._tracing import private_span


class TwilioWebhook(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(repr=False)
    params: dict[str, str] = Field(repr=False)
    signature: str = Field(repr=False)


def validate_twilio_webhook(webhook: TwilioWebhook, settings: Settings | None = None) -> None:
    with private_span("ingest.twilio.validate"):
        settings = settings or get_settings()
        token = settings.twilio_auth_token
        if token is None or not token.get_secret_value().strip():
            raise ConvIntelError("Configure TWILIO_AUTH_TOKEN before accepting webhooks.")
        try:
            url = urlsplit(webhook.url)
            form_url = (
                url.scheme in {"http", "https"}
                and bool(url.hostname)
                and not url.fragment
                and not url.username
                and "bodySHA256" not in parse_qs(url.query, keep_blank_values=True)
            )
            if not form_url:
                raise ConvIntelError("Only form-urlencoded Twilio webhooks are supported.")
            valid = bool(webhook.signature) and RequestValidator(token.get_secret_value()).validate(
                webhook.url, webhook.params, webhook.signature
            )
        except (ValueError, TypeError):
            raise ConvIntelError("Invalid Twilio webhook signature.") from None
        if not valid:
            raise ConvIntelError("Invalid Twilio webhook signature.")
        if (
            settings.twilio_account_sid is not None
            and webhook.params.get("AccountSid") != settings.twilio_account_sid
        ):
            raise ConvIntelError("Unexpected Twilio account.")
