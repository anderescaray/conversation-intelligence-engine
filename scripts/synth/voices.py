import shutil
import urllib.request
from pathlib import Path

from convintel.errors import ConvIntelError

from .schema import Record
from .storage import file_hash

REVISION = "c10ece1aade47bb51c153c893d14e5bf8e5b7117"
BASE = f"https://huggingface.co/rhasspy/piper-voices/resolve/{REVISION}/en/en_US"


class Voice(Record):
    model_id: str
    revision: str = REVISION
    speaker_id: int | None = None
    model_sha256: str
    config_sha256: str
    dataset_license: str
    model_license: str = "MIT (rhasspy/piper-voices repository metadata)"
    model_card: str
    directory: str


CUSTOMER = Voice(
    model_id="en_US-ljspeech-high",
    model_sha256="5d4f08ba6a2a48c44592eed3ce56bf85e9de3dd4e20df90541ae68a8310c029a",
    config_sha256="7e1f4634af596d83cca997fb7a931ba80b70f8a316a2655ee69c55365e0ace14",
    dataset_license="Public domain (LJ Speech)",
    directory="ljspeech/high",
    model_card=f"{BASE}/ljspeech/high/MODEL_CARD",
)
AGENT = Voice(
    model_id="en_US-libritts-high",
    speaker_id=0,
    model_sha256="9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f",
    config_sha256="2efdc6d7f954588b8180132cbd9b8001933fdd00932c92bc92fd0d2028a9eb3d",
    dataset_license="CC BY 4.0 (LibriTTS, speaker p3922)",
    directory="libritts/high",
    model_card=f"{BASE}/libritts/high/MODEL_CARD",
)


def verify_voice(directory: Path, voice: Voice) -> Path:
    model = directory / f"{voice.model_id}.onnx"
    config = directory / f"{voice.model_id}.onnx.json"
    for path, expected in ((model, voice.model_sha256), (config, voice.config_sha256)):
        if not path.is_file() or file_hash(path) != expected:
            raise ConvIntelError(
                "Voice assets missing or hash mismatch; explicitly provision voices"
            )
    return model


def provision(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for voice in (CUSTOMER, AGENT):
        for suffix, expected in (
            (".onnx", voice.model_sha256),
            (".onnx.json", voice.config_sha256),
        ):
            name = voice.model_id + suffix
            path = directory / name
            if path.exists():
                if file_hash(path) != expected:
                    raise ConvIntelError(
                        "Existing voice file has a different hash; use a new directory"
                    )
                continue
            temporary = path.with_suffix(path.suffix + ".part")
            with temporary.open("xb") as output:
                try:
                    with urllib.request.urlopen(
                        f"{BASE}/{voice.directory}/{name}", timeout=60
                    ) as source:
                        shutil.copyfileobj(source, output)
                except OSError:
                    raise ConvIntelError(
                        "Voice download failed; no unverified model will be loaded"
                    ) from None
            if file_hash(temporary) != expected:
                raise ConvIntelError("Downloaded voice hash mismatch; model was not installed")
            temporary.replace(path)
