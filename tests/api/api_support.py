"""Fakes and helpers shared by the API tests. All content is synthetic."""

import wave
from pathlib import Path


def write_wav(path: Path, seconds: float = 0.5, channels: int = 1, rate: int = 8000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\x00\x00" * channels * int(seconds * rate))
    return path
