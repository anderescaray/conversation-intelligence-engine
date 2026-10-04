import runpy
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest


@pytest.fixture
def generate() -> Callable[[Path], None]:
    script = Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "generate.py"
    return cast(Callable[[Path], None], runpy.run_path(str(script))["generate"])


def test_synthetic_fixture_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generate: Callable[[Path], None]
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "synthetic-espeak")

    def synthesize(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        with wave.open(args[args.index("-w") + 1], "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(22050)
            output.writeframes(b"\x01\x00" * 22050)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", synthesize)
    destination = tmp_path / "speech.wav"
    generate(destination)
    with wave.open(str(destination), "rb") as source:
        assert source.getnchannels() == 2
        assert source.getsampwidth() == 2
        assert source.getnframes() / source.getframerate() == 10
        assert source.readframes(1) == b"\x01\x00\x00\x00"
        source.setpos(source.getnframes() - 1)
        assert source.readframes(1) == b"\x00\x00\x01\x00"
    with pytest.raises(FileExistsError):
        generate(destination)


def test_fixture_requires_explicit_offline_synthesizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generate: Callable[[Path], None]
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="espeak"):
        generate(tmp_path / "speech.wav")
