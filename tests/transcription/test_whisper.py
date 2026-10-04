import asyncio
import math
import threading
import wave
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from convintel.contracts import Channel, Speaker, Transcriber
from convintel.errors import TranscriptionError
from convintel.transcription import WhisperSettings, WhisperTranscriber
from convintel.transcription import whisper as implementation


def segment(
    start: float, end: float, text: str = "Synthetic speech.", probability: float = 0.8
) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, text=text, avg_logprob=math.log(probability))


def result(
    *segments: SimpleNamespace, language: str = "en"
) -> tuple[Iterator[SimpleNamespace], SimpleNamespace]:
    return iter(segments), SimpleNamespace(language=language)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> Mock:
    factory = Mock()
    factory.return_value.transcribe.side_effect = [
        result(segment(2, 3, "Synthetic customer.", 0.9)),
        result(segment(0, 1, "Synthetic agent.", 0.6)),
    ]
    monkeypatch.setattr(implementation, "WhisperModel", factory)
    return factory


async def test_stereo_is_split_sorted_and_measured(
    audio: Path,
    model_path: Path,
    backend: Mock,
    spans: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(implementation, "perf_counter", Mock(side_effect=[100.0, 104.0]))
    transcriber: Transcriber = WhisperTranscriber(WhisperSettings(model_path=model_path))
    conversation = await transcriber.transcribe(str(audio), "synthetic-call")
    assert conversation.channel is Channel.VOICE
    assert not conversation.redacted
    assert conversation.id == "synthetic-call"
    assert conversation.audio_uri == str(audio)
    assert [turn.speaker for turn in conversation.turns] == [Speaker.AGENT, Speaker.CUSTOMER]
    assert [turn.index for turn in conversation.turns] == [0, 1]
    assert [turn.start_s for turn in conversation.turns] == [0, 2]
    assert [turn.asr_confidence for turn in conversation.turns] == pytest.approx([0.6, 0.9])
    backend.assert_called_once_with(
        str(model_path), device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True
    )
    calls = backend.return_value.transcribe.call_args_list
    assert len(calls) == 2
    assert calls[0].args[0].shape == (160000,)
    assert calls[1].args[0].shape == (160000,)
    assert np.max(np.abs(calls[0].args[0])) == pytest.approx(12000 / 32768, abs=0.001)
    assert np.max(np.abs(calls[1].args[0])) == pytest.approx(6000 / 32768, abs=0.001)
    assert calls[0].kwargs["vad_filter"] is True
    (span,) = spans.get_finished_spans()
    assert span.name == "asr.transcribe"
    assert span.attributes is not None
    assert span.attributes["audio.duration_s"] == 10
    assert span.attributes["asr.model"] == "small"
    assert float(conversation.metadata["asr_realtime_factor"]) == pytest.approx(0.4)
    assert span.attributes["asr.realtime_factor"] == float(
        conversation.metadata["asr_realtime_factor"]
    )
    assert "Synthetic customer" not in str(span.attributes)
    assert str(audio) not in str(span.attributes)


async def test_mono_has_unknown_speaker(audio: Path, model_path: Path, backend: Mock) -> None:
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\x01\x00" * 8000)
    backend.return_value.transcribe.side_effect = [result(segment(0, 0.5))]
    conversation = await WhisperTranscriber(WhisperSettings(model_path=model_path)).transcribe(
        audio.as_uri(), "mono"
    )
    assert conversation.turns[0].speaker is Speaker.UNKNOWN
    assert backend.return_value.transcribe.call_count == 1
    assert backend.return_value.transcribe.call_args.args[0].shape == (16000,)


async def test_custom_speaker_mapping(audio: Path, model_path: Path, backend: Mock) -> None:
    options = WhisperSettings(
        model_path=model_path, stereo_speakers=(Speaker.AGENT, Speaker.CUSTOMER)
    )
    conversation = await WhisperTranscriber(options).transcribe(str(audio), "outbound")
    assert [turn.speaker for turn in conversation.turns] == [Speaker.CUSTOMER, Speaker.AGENT]


async def test_empty_segments_and_silent_channel(
    audio: Path, model_path: Path, backend: Mock
) -> None:
    backend.return_value.transcribe.side_effect = [
        result(segment(0, 1, " ")),
        result(segment(1, 2), language="es"),
    ]
    conversation = await WhisperTranscriber(WhisperSettings(model_path=model_path)).transcribe(
        str(audio), "silent-left"
    )
    assert len(conversation.turns) == 1
    assert conversation.turns[0].speaker is Speaker.AGENT
    assert conversation.language == "es"


async def test_silence_returns_empty_conversation(
    audio: Path, model_path: Path, backend: Mock
) -> None:
    backend.return_value.transcribe.side_effect = [result(), result()]
    conversation = await WhisperTranscriber(WhisperSettings(model_path=model_path)).transcribe(
        str(audio), "silent"
    )
    assert conversation.turns == ()


async def test_native_decoder_is_allowlisted(
    audio: Path, model_path: Path, backend: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoder = Mock(wraps=implementation.av.open)
    monkeypatch.setattr(implementation.av, "open", decoder)
    await WhisperTranscriber(WhisperSettings(model_path=model_path)).transcribe(str(audio), "local")
    assert decoder.call_args.kwargs["options"] == {
        "format_whitelist": "wav,mp3,flac,ogg",
        "protocol_whitelist": "file",
    }


async def test_cuda_success_does_not_fallback(audio: Path, model_path: Path, backend: Mock) -> None:
    options = WhisperSettings(model_path=model_path, device="cuda", language="es")
    conversation = await WhisperTranscriber(options).transcribe(str(audio), "cuda")
    assert conversation.metadata["asr_device"] == "cuda"
    assert backend.call_count == 1
    assert backend.call_args.kwargs["compute_type"] == "float16"
    assert backend.return_value.transcribe.call_args.kwargs["language"] == "es"


async def test_model_is_lazy_and_cached(audio: Path, model_path: Path, backend: Mock) -> None:
    transcriber = WhisperTranscriber(WhisperSettings(model_path=model_path))
    backend.assert_not_called()
    backend.return_value.transcribe.side_effect = lambda *args, **kwargs: result(segment(0, 1))
    await transcriber.transcribe(str(audio), "one")
    await transcriber.transcribe(str(audio), "two")
    assert backend.call_count == 1


async def test_cache_resolution_is_offline(
    audio: Path, model_path: Path, backend: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolver = Mock(return_value=str(model_path))
    monkeypatch.setattr(implementation, "download_model", resolver)
    await WhisperTranscriber(WhisperSettings(model_size="tiny")).transcribe(str(audio), "cached")
    resolver.assert_called_once_with(
        "tiny", local_files_only=True, cache_dir=None, use_auth_token=False
    )


async def test_missing_cache_is_actionable(
    audio: Path, backend: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        implementation, "download_model", Mock(side_effect=FileNotFoundError("private-path"))
    )
    with pytest.raises(TranscriptionError, match="download_model") as error:
        await WhisperTranscriber().transcribe(str(audio), "no-model")
    assert "private-path" not in str(error.value)
    backend.assert_not_called()


@pytest.mark.parametrize(
    "missing",
    ["tokenizer.json", "model.bin", "config.json", "preprocessor_config.json", "vocabulary.txt"],
)
async def test_incomplete_model_never_reaches_backend(
    audio: Path, model_path: Path, backend: Mock, missing: str
) -> None:
    (model_path / missing).unlink()
    with pytest.raises(TranscriptionError, match="Download"):
        await WhisperTranscriber(WhisperSettings(model_path=model_path)).transcribe(
            str(audio), "incomplete"
        )
    backend.assert_not_called()


@pytest.mark.parametrize(
    "uri", ["https://example.test/audio.wav", "file://remote/audio.wav", "missing.wav"]
)
async def test_nonlocal_or_missing_audio(uri: str, model_path: Path, backend: Mock) -> None:
    with pytest.raises(TranscriptionError):
        await WhisperTranscriber(WhisperSettings(model_path=model_path)).transcribe(uri, "bad-uri")
    backend.assert_not_called()


async def test_invalid_audio(audio: Path, model_path: Path, backend: Mock) -> None:
    await asyncio.to_thread(audio.write_bytes, b"not an audio file")
    with pytest.raises(TranscriptionError):
        await WhisperTranscriber(WhisperSettings(model_path=model_path)).transcribe(
            str(audio), "bad-audio"
        )
    backend.assert_not_called()


@pytest.mark.parametrize("channels", [1, 3])
async def test_empty_or_multichannel_audio(
    audio: Path, model_path: Path, backend: Mock, channels: int
) -> None:
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"" if channels == 1 else b"\x00\x00" * channels * 16000)
    with pytest.raises(TranscriptionError):
        await WhisperTranscriber(WhisperSettings(model_path=model_path)).transcribe(
            str(audio), "invalid"
        )
    backend.assert_not_called()


async def test_lazy_inference_errors_are_safe(
    audio: Path, model_path: Path, backend: Mock, spans: InMemorySpanExporter
) -> None:
    def broken() -> Iterator[SimpleNamespace]:
        yield segment(0, 1)
        raise RuntimeError("private transcript or path")

    backend.return_value.transcribe.side_effect = None
    backend.return_value.transcribe.return_value = (broken(), SimpleNamespace(language="en"))
    with pytest.raises(TranscriptionError) as error:
        await WhisperTranscriber(WhisperSettings(model_path=model_path)).transcribe(
            str(audio), "failure"
        )
    assert "private transcript" not in str(error.value)
    (span,) = spans.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert not span.events
    assert not span.status.description


@pytest.mark.parametrize("failure", ["load", "compute_type", "iteration"])
async def test_cuda_errors_retry_all_channels_on_cpu(
    audio: Path, model_path: Path, backend: Mock, spans: InMemorySpanExporter, failure: str
) -> None:
    cpu = backend.return_value
    if failure == "load":
        backend.side_effect = [RuntimeError("unsupported sm_120"), cpu]
    elif failure == "compute_type":
        backend.side_effect = [ValueError("unsupported float16"), cpu]
    else:
        gpu = Mock()

        def broken() -> Iterator[SimpleNamespace]:
            yield segment(0, 1)
            raise RuntimeError("CUDA kernel failed")

        gpu.transcribe.return_value = (broken(), SimpleNamespace(language="en"))
        backend.side_effect = [gpu, cpu]
    conversation = await WhisperTranscriber(
        WhisperSettings(model_path=model_path, device="cuda")
    ).transcribe(str(audio), "fallback")
    assert len(conversation.turns) == 2
    assert backend.call_args_list[0].kwargs["device"] == "cuda"
    assert backend.call_args_list[1].kwargs["device"] == "cpu"
    assert backend.call_args_list[1].kwargs["compute_type"] == "int8"
    assert conversation.metadata["asr_device"] == "cpu"
    (span,) = spans.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["asr.fallback"] == "cpu_int8"


@pytest.mark.parametrize("field", ["start", "end", "avg_logprob"])
async def test_invalid_segments(audio: Path, model_path: Path, backend: Mock, field: str) -> None:
    bad = segment(0, 1)
    setattr(bad, field, math.nan)
    backend.return_value.transcribe.side_effect = [result(bad)]
    with pytest.raises(TranscriptionError):
        await WhisperTranscriber(WhisperSettings(model_path=model_path)).transcribe(
            str(audio), "nan"
        )


async def test_inference_does_not_block_event_loop(
    audio: Path, model_path: Path, backend: Mock
) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow(*args: object, **kwargs: object) -> tuple[Iterator[SimpleNamespace], SimpleNamespace]:
        started.set()
        assert release.wait(5)
        return result(segment(0, 1))

    backend.return_value.transcribe.side_effect = slow
    task = asyncio.create_task(
        WhisperTranscriber(WhisperSettings(model_path=model_path)).transcribe(
            str(audio), "threaded"
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        assert not task.done()
    finally:
        release.set()
    await task
