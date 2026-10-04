import argparse
import wave
from pathlib import Path

SOURCE = Path(__file__).with_name("jfk.wav")


def generate(destination: Path) -> None:
    with wave.open(str(SOURCE), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            raise ValueError("The JFK fixture must be mono PCM16 at 16 kHz.")
        if source.getnframes() < 11 * 16000:
            raise ValueError("The JFK fixture must contain at least eleven seconds.")
        left = source.readframes(5 * 16000) + bytes(5 * 16000 * 2)
        source.setpos(6 * 16000)
        right = bytes(5 * 16000 * 2) + source.readframes(5 * 16000)
    stereo = bytearray(10 * 16000 * 4)
    for index, pcm in enumerate((left, right)):
        stereo[index * 2 :: 4] = pcm[0::2]
        stereo[index * 2 + 1 :: 4] = pcm[1::2]
    with destination.open("xb") as output, wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(stereo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ten seconds of stereo speech from JFK.")
    parser.add_argument("destination", type=Path)
    generate(parser.parse_args().destination)
