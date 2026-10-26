"""Twilio webhooks, wired to the T01 normalizers and their signature check.

Signature verification runs once on its own so that a failure maps to 403 without parsing
exception messages; the normalizers then re-verify (they have no bypass) and any error left is
a malformed payload (400). Neither response, span nor log carries a form value.
"""

import re
from typing import Annotated
from urllib.parse import parse_qsl

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from convintel.api.audio import AudioSpool
from convintel.api.dependencies import (
    ApiState,
    State,
    get_app_settings,
    get_fetcher,
    get_pipeline,
    get_spool,
    get_transcriber,
)
from convintel.api.errors import ServiceNotConfiguredError
from convintel.api.middleware import api_span
from convintel.api.protocols import Pipeline, RecordingFetcher
from convintel.api.routes import transcribe_spooled, unredacted
from convintel.config import Settings
from convintel.contracts import Transcriber
from convintel.errors import ConvIntelError
from convintel.ingest.normalizers import normalize_twilio_recording, normalize_twilio_sms
from convintel.ingest.twilio import TwilioWebhook, validate_twilio_webhook

# Real Twilio callbacks are a few KB; the cap bounds memory before signature checks.
MAX_FORM_BYTES = 64 * 1024
MAX_FORM_FIELDS = 200
EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'

_ACCOUNT_SID = re.compile(r"^AC[0-9a-fA-F]{32}$")
_FORM_TYPE = "application/x-www-form-urlencoded"

router = APIRouter(prefix="/v1/twilio")
AppSettings = Annotated[Settings, Depends(get_app_settings)]


def _require_twilio_config(settings: Settings) -> None:
    """Fail closed unless both credentials are set (issue #17: account SID is mandatory)."""
    token = settings.twilio_auth_token
    sid = settings.twilio_account_sid
    if token is None or not token.get_secret_value().strip() or not _ACCOUNT_SID.match(sid or ""):
        raise ServiceNotConfiguredError("Twilio webhooks need TWILIO_AUTH_TOKEN and ACCOUNT_SID.")


async def _read_form(request: Request) -> dict[str, str]:
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != _FORM_TYPE:
        raise HTTPException(415, "Expected application/x-www-form-urlencoded.")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_FORM_BYTES:
        raise HTTPException(413, "Request body too large.")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_FORM_BYTES:
            raise HTTPException(413, "Request body too large.")
    try:
        pairs = parse_qsl(
            body.decode("utf-8"), keep_blank_values=True, max_num_fields=MAX_FORM_FIELDS
        )
    except (UnicodeDecodeError, ValueError):
        raise HTTPException(400, "Malformed form body.") from None
    params = dict(pairs)
    # Every field is signed; collapsing duplicates would validate a different payload.
    if len(params) != len(pairs):
        raise HTTPException(400, "Repeated form fields are not accepted.")
    return params


def _public_url(request: Request, public_base_url: str | None) -> str:
    """The URL Twilio signed: configured public base + path, never forwarded headers."""
    if public_base_url is None:
        return str(request.url)
    query = f"?{request.url.query}" if request.url.query else ""
    return f"{public_base_url.rstrip('/')}{request.url.path}{query}"


async def _verified_webhook(request: Request, state: ApiState, settings: Settings) -> TwilioWebhook:
    _require_twilio_config(settings)
    webhook = TwilioWebhook(
        url=_public_url(request, state.public_base_url),
        params=await _read_form(request),
        signature=request.headers.get("x-twilio-signature", ""),
    )
    try:
        validate_twilio_webhook(webhook, settings)
    except ConvIntelError:
        raise HTTPException(403, "Invalid Twilio webhook.") from None
    return webhook


@router.post("/sms")
async def twilio_sms(
    request: Request,
    state: State,
    settings: AppSettings,
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
) -> Response:
    with api_span("api.twilio.sms") as span:
        webhook = await _verified_webhook(request, state, settings)
        try:
            conversation = normalize_twilio_sms(webhook, settings=settings)
        except ConvIntelError:
            raise HTTPException(400, "Invalid Twilio SMS payload.") from None
        span.set_attribute("conversation.id", conversation.id)
        span.set_attribute("channel", conversation.channel.value)
        await pipeline.analyze(unredacted(conversation))
        return Response(EMPTY_TWIML, media_type="application/xml")


@router.post("/recording", status_code=204)
async def twilio_recording(
    request: Request,
    state: State,
    settings: AppSettings,
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
    spool: Annotated[AudioSpool, Depends(get_spool)],
    transcriber: Annotated[Transcriber, Depends(get_transcriber)],
    fetcher: Annotated[RecordingFetcher, Depends(get_fetcher)],
) -> Response:
    with api_span("api.twilio.recording") as span:
        webhook = await _verified_webhook(request, state, settings)
        try:
            recording = normalize_twilio_recording(webhook, settings=settings)
        except ConvIntelError:
            raise HTTPException(400, "Invalid Twilio recording payload.") from None
        span.set_attribute("conversation.id", recording.id)
        span.set_attribute("channel", recording.channel.value)
        # Reject from the declared duration before downloading anything.
        spool.check_duration(float(recording.metadata["audio_duration_s"]))
        local = await fetcher.fetch(recording, spool.root)
        conversation = await transcribe_spooled(recording, str(local), spool, transcriber)
        await pipeline.analyze(unredacted(conversation))
        return Response(status_code=204)
