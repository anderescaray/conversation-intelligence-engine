import argparse
import shutil
import subprocess
import wave
from pathlib import Path
from tempfile import TemporaryDirectory


def generate(destination: Path) -> None:
    executable = shutil.which("espeak-ng") or shutil.which("espeak")
    if executable is None:
        raise RuntimeError("Synthetic speech generation requires local espeak-ng or espeak.")
    phrases = (
        "Hello. I need help with my test order. It has not arrived.",
        "Hello. I can check your test order and help you today.",
    )
    with TemporaryDirectory() as temporary:
        clips: list[bytes] = []
        rate = 0
        for index, text in enumerate(phrases):
            path = Path(temporary) / f"speaker-{index}.wav"
            subprocess.run(
                [executable, "-v", "en", "-s", "180", "-w", str(path), text],
                check=True,
                capture_output=True,
            )
            with wave.open(str(path), "rb") as source:
                if source.getnchannels() != 1 or source.getsampwidth() != 2:
                    raise ValueError("Speech synthesizer must produce mono PCM16 WAV.")
                if rate and rate != source.getframerate():
                    raise ValueError("Speech clips must share a sample rate.")
                rate = source.getframerate()
                pcm = source.readframes(source.getnframes())
            if len(pcm) > rate * 5 * 2:
                raise ValueError("Synthetic phrase exceeds its five-second channel window.")
            padding = bytes(rate * 10 * 2 - len(pcm))
            clips.append(pcm + padding if index == 0 else padding + pcm)
        stereo = bytearray(rate * 10 * 4)
        for index, pcm in enumerate(clips):
            stereo[index * 2 :: 4] = pcm[0::2]
            stereo[index * 2 + 1 :: 4] = pcm[1::2]
        with destination.open("xb") as output, wave.open(output, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(stereo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ten seconds of synthetic stereo speech.")
    parser.add_argument("destination", type=Path)
    generate(parser.parse_args().destination)
