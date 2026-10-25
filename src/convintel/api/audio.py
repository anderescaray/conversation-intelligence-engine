"""Gate between an `audio_uri` and the transcriber (issue #17).

The transcriber decodes the whole file to float32 PCM in memory, so anything that reaches it
must be a regular file inside the spool directory, small enough on disk and short enough by its
container header. Every check happens before a single frame is decoded.
"""

import stat
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

import av
from av.error import FFmpegError

from convintel.errors import ConvIntelError

# Twilio dual-channel 8 kHz 16-bit WAV is ~115 MB per hour; one hour is the ceiling.
MAX_AUDIO_BYTES = 128 * 1024 * 1024
MAX_AUDIO_DURATION_S = 3600.0
SPOOL_DIR = Path("var/spool/audio")

# Same demuxers and protocol the transcriber allows, so probing cannot reach further than decoding.
_PROBE_OPTIONS = {"format_whitelist": "wav,mp3,flac,ogg", "protocol_whitelist": "file"}

AudioRejection = Literal["outside_spool", "not_found", "too_large", "too_long", "unreadable"]


class AudioRejectedError(ConvIntelError):
    """The audio failed a spool check. `reason` is a fixed code, safe to return to clients."""

    def __init__(self, reason: AudioRejection) -> None:
        super().__init__(reason)
        self.reason: AudioRejection = reason


class AudioSpool:
    def __init__(
        self,
        root: Path = SPOOL_DIR,
        *,
        max_bytes: int = MAX_AUDIO_BYTES,
        max_duration_s: float = MAX_AUDIO_DURATION_S,
    ) -> None:
        self.root = root.resolve()
        self.max_bytes = max_bytes
        self.max_duration_s = max_duration_s

    def check_duration(self, duration_s: float) -> None:
        if duration_s > self.max_duration_s:
            raise AudioRejectedError("too_long")

    def resolve(self, audio_uri: str) -> Path:
        """Return the real path of a spooled audio file, or raise `AudioRejectedError`.

        Symlinks are resolved before the confinement check, so a link inside the spool that
        points outside it is rejected like a `..` path.
        """
        uri = urlsplit(audio_uri)
        if (
            uri.scheme not in {"", "file"}
            or uri.netloc not in {"", "localhost"}
            or uri.query
            or uri.fragment
        ):
            raise AudioRejectedError("outside_spool")
        raw = Path(unquote(uri.path) if uri.scheme else audio_uri)
        # Confinement is checked before existence so responses never reveal what exists
        # outside the spool.
        try:
            path = (raw if raw.is_absolute() else self.root / raw).resolve()
        except (OSError, RuntimeError):
            raise AudioRejectedError("outside_spool") from None
        if not path.is_relative_to(self.root):
            raise AudioRejectedError("outside_spool")
        try:
            path = path.resolve(strict=True)
            info = path.stat()
        except (OSError, RuntimeError):
            raise AudioRejectedError("not_found") from None
        if not path.is_relative_to(self.root):
            raise AudioRejectedError("outside_spool")
        if not stat.S_ISREG(info.st_mode):
            raise AudioRejectedError("not_found")
        if info.st_size > self.max_bytes:
            raise AudioRejectedError("too_large")
        duration = _probe_duration_s(path)
        if duration is None:
            raise AudioRejectedError("unreadable")
        self.check_duration(duration)
        return path


def _probe_duration_s(path: Path) -> float | None:
    """Duration from container/stream headers only; `None` when the file does not declare one."""
    try:
        with path.open("rb") as audio, av.open(audio, mode="r", options=_PROBE_OPTIONS) as box:
            if box.duration is not None:
                return box.duration / av.time_base
            streams = box.streams.audio
            if streams and streams[0].duration is not None and streams[0].time_base is not None:
                return float(streams[0].duration * streams[0].time_base)
    except (OSError, ValueError, FFmpegError):
        return None
    return None
