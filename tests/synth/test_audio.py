import sys
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from convintel.contracts import Speaker
from convintel.errors import ConvIntelError
from scripts.synth.audio import GAP_FRAMES, RATE, PiperRenderer, render_conversation
from scripts.synth.scenarios import scenarios
from scripts.synth.storage import file_hash


class FakeRenderer:
    def speak(self, text: str, speaker: Speaker) -> bytes:
        return np.full(160, 200 if speaker == Speaker.CUSTOMER else 400, dtype="<i2").tobytes()


def test_stereo_and_timestamps(
    tmp_path: Path,
    sample_factory: Callable,
    spans: InMemorySpanExporter,
) -> None:
    sample = sample_factory(scenarios(42)[2])
    path = tmp_path / "voice.wav"
    conversation, manifest = render_conversation(sample.conversation, FakeRenderer(), path)
    assert manifest.sha256 == file_hash(path)
    with wave.open(str(path), "rb") as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (2, 2, RATE)
        data = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").reshape(-1, 2)
    for index, turn in enumerate(conversation.turns):
        start = index * (160 + GAP_FRAMES)
        assert turn.start_s == start / RATE
        assert turn.end_s == (start + 160) / RATE
        active = 0 if turn.speaker == Speaker.CUSTOMER else 1
        assert np.all(data[start : start + 160, active] != 0)
        assert np.all(data[start : start + 160, 1 - active] == 0)
        assert np.all(data[start + 160 : start + 160 + GAP_FRAMES] == 0)
        assert turn.text == sample.conversation.turns[index].text
    assert conversation.audio_uri == str(path.resolve())
    assert spans.get_finished_spans()[-1].name == "synthetic.tts"
    with pytest.raises(ConvIntelError):
        render_conversation(sample.conversation, FakeRenderer(), path)


def test_missing_voices_never_download_or_import_piper(tmp_path: Path) -> None:
    before = "piper" in sys.modules
    with pytest.raises(ConvIntelError, match="explicitly provision"):
        PiperRenderer(tmp_path)
    assert ("piper" in sys.modules) == before


def test_empty_tts_is_rejected(tmp_path: Path, sample_factory: Callable) -> None:
    class Empty:
        def speak(self, text: str, speaker: Speaker) -> bytes:
            return b""

    with pytest.raises(ConvIntelError, match="empty"):
        render_conversation(
            sample_factory(scenarios(42)[2]).conversation, Empty(), tmp_path / "empty.wav"
        )


def test_native_tts_errors_do_not_expose_text() -> None:
    class FailingVoice:
        def synthesize_wav(self, text: str, output: wave.Wave_write, config: object) -> None:
            raise RuntimeError(text)

    renderer = PiperRenderer.__new__(PiperRenderer)
    renderer._voices = {Speaker.CUSTOMER: FailingVoice()}
    renderer._configs = {Speaker.CUSTOMER: None}
    with pytest.raises(ConvIntelError, match="TTS failed") as error:
        renderer.speak("synthetic-private-value", Speaker.CUSTOMER)
    assert "synthetic-private-value" not in str(error.value)
    assert error.value.__suppress_context__
