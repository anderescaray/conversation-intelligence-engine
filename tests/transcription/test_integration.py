import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

from convintel.contracts import Speaker
from convintel.errors import TranscriptionError
from convintel.transcription import WhisperSettings, WhisperTranscriber

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tests" / "fixtures" / "audio" / "generate.py"


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
async def test_real_small_model_on_jfk_speech(
    tmp_path: Path, request: pytest.FixtureRequest, device: Literal["cpu", "cuda"]
) -> None:
    pytest.importorskip("faster_whisper")
    supplied_path = request.config.getoption("--asr-model-path")
    path = Path(supplied_path) if supplied_path else ROOT / "models" / "whisper-small"
    transcriber = WhisperTranscriber(WhisperSettings(model_path=path, device=device, language="en"))
    try:
        await asyncio.to_thread(transcriber._resolve_model)
    except TranscriptionError:
        if supplied_path:
            raise
        pytest.skip("Provision models/whisper-small explicitly; tests never download weights.")
    audio = tmp_path / "jfk-stereo.wav"
    await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(GENERATOR), str(audio)],
        check=True,
        capture_output=True,
    )
    for phase in ("cold", "warm"):
        conversation = await transcriber.transcribe(str(audio), f"jfk-integration-{device}-{phase}")
        assert {turn.speaker for turn in conversation.turns} == {Speaker.CUSTOMER, Speaker.AGENT}
        assert all(
            turn.text.strip() and turn.asr_confidence is not None for turn in conversation.turns
        )
        assert float(conversation.metadata["audio_duration_s"]) == pytest.approx(10)
        actual = conversation.metadata["asr_device"]
        assert actual in ({"cpu"} if device == "cpu" else {"cpu", "cuda"})
        print(
            f"small, requested={device}, actual={actual}, 4 threads, language=en, "
            f"{phase} RTF={conversation.metadata['asr_realtime_factor']}"
        )
