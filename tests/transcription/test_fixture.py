import hashlib
import runpy
import wave
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"


@pytest.fixture
def generate() -> Callable[[Path], None]:
    return cast(Callable[[Path], None], runpy.run_path(str(FIXTURES / "generate.py"))["generate"])


def test_jfk_fixture_integrity() -> None:
    assert hashlib.sha256((FIXTURES / "jfk.wav").read_bytes()).hexdigest() == (
        "b9e1ae4e0837e7b99f05e4f61f70f5732320a56614ab4514d803fa85f9a563c4"
    )


def test_stereo_fixture_uses_different_source_segments(
    tmp_path: Path, generate: Callable[[Path], None]
) -> None:
    destination = tmp_path / "speech.wav"
    generate(destination)
    with wave.open(str(FIXTURES / "jfk.wav"), "rb") as source:
        left = source.readframes(5 * 16000)
        source.setpos(6 * 16000)
        right = source.readframes(5 * 16000)
    with wave.open(str(destination), "rb") as stereo:
        assert stereo.getnchannels() == 2
        assert stereo.getsampwidth() == 2
        assert stereo.getnframes() / stereo.getframerate() == 10
        pcm = stereo.readframes(stereo.getnframes())
    expected = (left + bytes(5 * 16000 * 2), bytes(5 * 16000 * 2) + right)
    for index, channel in enumerate(expected):
        assert pcm[index * 2 :: 4] == channel[0::2]
        assert pcm[index * 2 + 1 :: 4] == channel[1::2]
    assert left != right
    with pytest.raises(FileExistsError):
        generate(destination)
