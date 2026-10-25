from collections.abc import Callable, Iterator
from pathlib import Path
from urllib.parse import urlencode

import httpx2
import pytest
from api_support import ACCOUNT, TOKEN, FakeFetcher, FakePipeline, FakeTranscriber, sign, write_wav
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from convintel.api.twilio import EMPTY_TWIML, MAX_FORM_BYTES
from convintel.config import Settings

BASE = "http://testserver"
SMS = "/v1/twilio/sms"
RECORDING = "/v1/twilio/recording"
FORM = {"content-type": "application/x-www-form-urlencoded"}


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
        "RecordingUrl": f"https://api.twilio.com/2010-04-01/Accounts/{ACCOUNT}/Recordings/RE"
        + "3" * 32,
        "RecordingStatus": "completed",
        "RecordingDuration": "1",
        "RecordingChannels": "1",
    }


@pytest.fixture
def fetcher() -> FakeFetcher:
    return FakeFetcher(lambda spool: write_wav(spool / "RE-synthetic.wav", seconds=1))


@pytest.fixture
def twilio(make_app: Callable[..., FastAPI], fetcher: FakeFetcher) -> Iterator[TestClient]:
    with TestClient(make_app(fetcher=fetcher)) as client:
        yield client


def post(
    client: TestClient,
    path: str,
    params: dict[str, str],
    *,
    url: str | None = None,
    signature: str | None = None,
    body: str | None = None,
) -> httpx2.Response:
    signature = signature if signature is not None else sign(url or BASE + path, params)
    return client.post(
        path,
        content=body if body is not None else urlencode(params),
        headers={**FORM, "x-twilio-signature": signature},
    )


def test_sms_is_verified_normalized_and_analyzed(
    twilio: TestClient, pipeline: FakePipeline, sms: dict[str, str], spans: InMemorySpanExporter
) -> None:
    response = post(twilio, SMS, sms)
    assert response.status_code == 200
    assert response.text == EMPTY_TWIML
    (conversation,) = pipeline.seen
    assert conversation.id == sms["MessageSid"]
    assert conversation.turns[0].text == sms["Body"]
    (span,) = [s for s in spans.get_finished_spans() if s.name == "api.twilio.sms"]
    assert dict(span.attributes or {}) == {"conversation.id": sms["MessageSid"], "channel": "sms"}


@pytest.mark.parametrize("tamper", ["signature", "body", "token"])
def test_rejects_bad_signatures(
    twilio: TestClient, pipeline: FakePipeline, sms: dict[str, str], tamper: str
) -> None:
    if tamper == "signature":
        response = post(twilio, SMS, sms, signature="")
    elif tamper == "body":
        response = post(twilio, SMS, {**sms, "Body": "changed"}, signature=sign(BASE + SMS, sms))
    else:
        response = post(twilio, SMS, sms, signature=sign(BASE + SMS, sms, token="other"))
    assert response.status_code == 403
    assert response.json() == {"detail": "Invalid Twilio webhook."}
    assert not pipeline.seen


def test_rejects_another_account(twilio: TestClient, sms: dict[str, str]) -> None:
    response = post(twilio, SMS, {**sms, "AccountSid": "AC" + "9" * 32})
    assert response.status_code == 403


@pytest.mark.parametrize(
    "creds",
    [
        {"twilio_auth_token": SecretStr(TOKEN)},
        {"twilio_auth_token": SecretStr(TOKEN), "twilio_account_sid": "not-a-sid"},
        {"twilio_account_sid": ACCOUNT},
    ],
)
def test_fails_closed_without_twilio_config(
    make_app: Callable[..., FastAPI],
    pipeline: FakePipeline,
    sms: dict[str, str],
    creds: dict[str, object],
) -> None:
    settings = Settings.model_validate({"_env_file": None, **creds})
    with TestClient(make_app(settings=settings)) as client:
        response = post(client, SMS, sms)
    assert response.status_code == 503
    assert response.json() == {"detail": "Service not configured."}
    assert not pipeline.seen


def test_signs_against_the_public_base_url(
    make_app: Callable[..., FastAPI], pipeline: FakePipeline, sms: dict[str, str]
) -> None:
    public = "https://api.example.test/hooks"
    with TestClient(make_app(public_base_url=public + "/")) as client:
        assert post(client, SMS, sms).status_code == 403
        ok = post(client, SMS, sms, url=public + SMS)
        assert ok.status_code == 200
    assert len(pipeline.seen) == 1


def test_rejects_repeated_fields(twilio: TestClient, sms: dict[str, str]) -> None:
    body = urlencode(sms) + "&Body=second"
    response = post(twilio, SMS, sms, body=body)
    assert response.status_code == 400
    assert "second" not in response.text


def test_rejects_oversized_bodies(twilio: TestClient, sms: dict[str, str]) -> None:
    big = {**sms, "Body": "x" * MAX_FORM_BYTES}
    assert post(twilio, SMS, big).status_code == 413


def test_rejects_other_content_types(twilio: TestClient, sms: dict[str, str]) -> None:
    response = twilio.post(SMS, json=sms, headers={"x-twilio-signature": "x"})
    assert response.status_code == 415


def test_signed_but_malformed_payload_is_400(
    twilio: TestClient, pipeline: FakePipeline, sms: dict[str, str]
) -> None:
    del sms["Body"]
    response = post(twilio, SMS, sms)
    assert response.status_code == 400
    assert not pipeline.seen


def test_recording_is_fetched_into_the_spool_and_transcribed(
    twilio: TestClient,
    pipeline: FakePipeline,
    transcriber: FakeTranscriber,
    fetcher: FakeFetcher,
    recording: dict[str, str],
    spool_dir: Path,
) -> None:
    response = post(twilio, RECORDING, recording)
    assert response.status_code == 204
    assert fetcher.calls == 1
    assert transcriber.calls == [
        (str((spool_dir / "RE-synthetic.wav").resolve()), recording["CallSid"])
    ]
    (conversation,) = pipeline.seen
    assert conversation.metadata["recording_sid"] == recording["RecordingSid"]
    assert conversation.metadata["asr_model"] == "fake"


def test_recording_too_long_is_rejected_before_fetching(
    twilio: TestClient, fetcher: FakeFetcher, recording: dict[str, str]
) -> None:
    response = post(twilio, RECORDING, {**recording, "RecordingDuration": "7200"})
    assert response.status_code == 413
    assert fetcher.calls == 0


def test_fetched_file_outside_the_spool_is_rejected(
    make_app: Callable[..., FastAPI],
    transcriber: FakeTranscriber,
    recording: dict[str, str],
    tmp_path: Path,
) -> None:
    escape = FakeFetcher(lambda spool: write_wav(tmp_path / "escape.wav"))
    with TestClient(make_app(fetcher=escape)) as client:
        response = post(client, RECORDING, recording)
    assert response.status_code == 422
    assert not transcriber.calls


def test_recording_without_a_fetcher_is_503(
    client: TestClient, pipeline: FakePipeline, recording: dict[str, str]
) -> None:
    response = post(client, RECORDING, recording)
    assert response.status_code == 503
    assert not pipeline.seen
