import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from convintel.contracts import Speaker
from convintel.errors import TranscriptionError
from convintel.transcription import WhisperSettings, WhisperTranscriber

GENERATOR = Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "generate.py"


@pytest.mark.integration
async def test_real_small_model_on_synthetic_speech(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    supplied_path = request.config.getoption("--asr-model-path")
    options = WhisperSettings(model_path=Path(supplied_path) if supplied_path else None)
    transcriber = WhisperTranscriber(options)
    try:
        await asyncio.to_thread(transcriber._resolve_model)
    except TranscriptionError:
        if supplied_path:
            raise
        pytest.skip("Preinstall the small model explicitly; no weights are downloaded by tests.")
    if not (shutil.which("espeak-ng") or shutil.which("espeak")):
        pytest.skip("Local espeak-ng or espeak is required to synthesize speech offline.")
    audio = tmp_path / "synthetic-speech.wav"
    await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(GENERATOR), str(audio)],
        check=True,
        capture_output=True,
    )
    for phase in ("cold", "warm"):
        conversation = await transcriber.transcribe(str(audio), "synthetic-integration")
        assert {turn.speaker for turn in conversation.turns} == {Speaker.CUSTOMER, Speaker.AGENT}
        assert all(
            turn.text.strip() and turn.asr_confidence is not None for turn in conversation.turns
        )
        assert float(conversation.metadata["audio_duration_s"]) == pytest.approx(10)
        assert conversation.metadata["asr_device"] == "cpu"
        print(
            f"small CPU int8, 4 threads, {phase} RTF={conversation.metadata['asr_realtime_factor']}"
        )
