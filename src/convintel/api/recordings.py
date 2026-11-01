"""Lifecycle of recordings the API downloads itself.

Twilio keeps the original and retries the webhook on failure, so a local copy is only needed
while it is transcribed: it is deleted afterwards whatever the outcome, and a retry refetches.
Files clients place in the spool are never deleted here (the API does not own them).
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from convintel.api.audio import AudioSpool
from convintel.api.protocols import RecordingFetcher
from convintel.contracts import Conversation


@asynccontextmanager
async def fetched_recording(
    fetcher: RecordingFetcher, recording: Conversation, spool: AudioSpool
) -> AsyncIterator[str]:
    """Fetch `recording` into the spool and yield its path; delete it on exit, always."""
    local = await fetcher.fetch(recording, spool.root)
    try:
        yield str(local)
    finally:
        await asyncio.to_thread(spool.discard, local)
