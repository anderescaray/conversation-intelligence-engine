"""Seams the API depends on but does not own.

The pipeline (T10) is built in another branch; the API codes against this Protocol so both can
merge in either order. The recording fetcher is a Protocol because downloading Twilio media
needs credentials and egress rules that part 1 does not decide.
"""

from pathlib import Path
from typing import Protocol

from convintel.contracts import Conversation, ConversationInsights


class Pipeline(Protocol):
    async def analyze(self, conversation: Conversation) -> ConversationInsights:
        """Redact and analyze one conversation. Raise `ConvIntelError` on expected failures."""
        ...

    async def warmup(self) -> None:
        """Load models and open clients. Raising aborts API startup."""
        ...


class RecordingFetcher(Protocol):
    async def fetch(self, recording: Conversation, spool_dir: Path) -> Path:
        """Store `recording.audio_uri` as a file under `spool_dir` and return its path.

        The API re-checks the returned path (confinement, size, duration) before transcribing,
        so implementations cannot widen what reaches the decoder.
        """
        ...
