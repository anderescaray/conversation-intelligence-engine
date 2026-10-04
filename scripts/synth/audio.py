import argparse
import importlib
import io
import time
import wave
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import av
import numpy as np

from convintel.contracts import Conversation, Speaker, Turn
from convintel.errors import ConvIntelError
from convintel.telemetry import traced

from .schema import VERSION, Record
from .storage import file_hash, read_jsonl, write_json, write_jsonl
from .voices import AGENT, CUSTOMER, Voice, provision, verify_voice

RATE = 16000
GAP_FRAMES = 4000


class Renderer(Protocol):
    def speak(self, text: str, speaker: Speaker) -> bytes: ...


class AudioFile(Record):
    conversation_id: str
    filename: str
    sha256: str
    frames: int
    duration_s: float
    render_s: float


class AudioManifest(Record):
    generator_version: str = VERSION
    source_sha256: str
    piper_version: str
    sample_rate: int = RATE
    sample_width: int = 2
    channels: tuple[str, str] = ("customer", "agent")
    gap_frames: int = GAP_FRAMES
    noise_scale: float = 0
    noise_w_scale: float = 0
    length_scale: float = 1
    customer_voice: Voice = CUSTOMER
    agent_voice: Voice = AGENT
    files: list[AudioFile]
    conversations_sha256: str


class PiperRenderer:
    def __init__(self, directory: Path) -> None:
        models = {Speaker.CUSTOMER: CUSTOMER, Speaker.AGENT: AGENT}
        paths = {speaker: verify_voice(directory, voice) for speaker, voice in models.items()}
        try:
            piper = importlib.import_module("piper")
        except ModuleNotFoundError:
            raise ConvIntelError("Piper is optional; install it with uv sync --group ml") from None
        self._voices = {
            speaker: piper.PiperVoice.load(path, use_cuda=False) for speaker, path in paths.items()
        }
        self._configs = {
            speaker: piper.SynthesisConfig(
                speaker_id=voice.speaker_id,
                noise_scale=0,
                noise_w_scale=0,
                length_scale=1,
            )
            for speaker, voice in models.items()
        }

    def speak(self, text: str, speaker: Speaker) -> bytes:
        if speaker not in self._voices:
            raise ConvIntelError("Only customer and agent voices are supported")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            try:
                self._voices[speaker].synthesize_wav(text, output, self._configs[speaker])
            except Exception as exc:
                raise ConvIntelError(f"TTS failed ({type(exc).__name__})") from None
        buffer.seek(0)
        with wave.open(buffer, "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise ConvIntelError("Piper must return mono PCM16")
            data = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(data, format="s16", layout="mono")
            frame.sample_rate = source.getframerate()
        resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
        return b"".join(
            frame.to_ndarray().astype("<i2").tobytes()
            for frame in [*resampler.resample(frame), *resampler.resample(None)]
        )


def render_conversation(
    conversation: Conversation,
    renderer: Renderer,
    path: Path,
) -> tuple[Conversation, AudioFile]:
    if conversation.channel != "voice" or not conversation.turns or path.exists():
        raise ConvIntelError("Audio requires a voice conversation and a new output file")
    if any(turn.speaker not in {Speaker.CUSTOMER, Speaker.AGENT} for turn in conversation.turns):
        raise ConvIntelError("Only customer and agent voices are supported")
    start = time.perf_counter()
    turns: list[Turn] = []
    cursor = 0
    with traced("synthetic.tts", version=VERSION) as span:
        span.set_attribute("conversation.id", conversation.id)
        with path.open("xb") as stream, wave.open(stream, "wb") as output:
            output.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
            for turn in conversation.turns:
                mono = renderer.speak(turn.text, turn.speaker)
                if not mono or len(mono) % 2:
                    raise ConvIntelError("TTS returned empty or invalid PCM16")
                frames = len(mono) // 2
                stereo = np.zeros((frames, 2), dtype="<i2")
                stereo[:, 0 if turn.speaker == Speaker.CUSTOMER else 1] = np.frombuffer(
                    mono, dtype="<i2"
                )
                output.writeframes(stereo.tobytes())
                turns.append(
                    turn.model_copy(
                        update={"start_s": cursor / RATE, "end_s": (cursor + frames) / RATE}
                    )
                )
                output.writeframes(bytes(GAP_FRAMES * 4))
                cursor += frames + GAP_FRAMES
        span.set_attribute("audio.duration_s", cursor / RATE)
    updated = conversation.model_copy(
        update={"turns": tuple(turns), "audio_uri": str(path.resolve())}
    )
    return updated, AudioFile(
        conversation_id=conversation.id,
        filename=path.name,
        sha256=file_hash(path),
        frames=cursor,
        duration_s=cursor / RATE,
        render_s=time.perf_counter() - start,
    )


def render_dataset(source: Path, destination: Path, voices: Path) -> AudioManifest:
    conversations = [c for c in read_jsonl(source, Conversation) if c.channel == "voice"]
    if not conversations or destination.exists():
        raise ConvIntelError("Need voice samples and a new output directory")
    if len({c.id for c in conversations}) != len(conversations):
        raise ConvIntelError("Duplicate conversation IDs")
    for conversation in conversations:
        if not conversation.id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in conversation.id
        ):
            raise ConvIntelError("Unsafe conversation ID")
    renderer = PiperRenderer(voices)
    destination.mkdir(parents=True)
    rendered: list[Conversation] = []
    files: list[AudioFile] = []
    for conversation in conversations:
        updated, audio_file = render_conversation(
            conversation, renderer, destination / f"{conversation.id}.wav"
        )
        rendered.append(updated)
        files.append(audio_file)
    write_jsonl(destination / "conversations.jsonl", rendered)
    result = AudioManifest(
        source_sha256=file_hash(source),
        piper_version=version("piper-tts"),
        files=files,
        conversations_sha256=file_hash(destination / "conversations.jsonl"),
    )
    write_json(destination / "manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Explicit local TTS; never calls an LLM")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--voices-dir", type=Path, default=Path("data/generated/voices"))
    parser.add_argument("--download-voices", action="store_true")
    args = parser.parse_args()
    try:
        if args.download_voices:
            provision(args.voices_dir)
        else:
            if args.dataset is None or args.output is None:
                parser.error("Rendering requires --dataset and --output")
            print(
                render_dataset(args.dataset, args.output, args.voices_dir).model_dump_json(indent=2)
            )
    except ConvIntelError as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
