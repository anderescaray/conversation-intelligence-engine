"""Client-facing analysis endpoint and operational probes."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from convintel.api.audio import AudioSpool
from convintel.api.dependencies import State, get_pipeline, get_spool, get_transcriber
from convintel.api.middleware import api_span
from convintel.api.protocols import Pipeline
from convintel.contracts import Conversation, ConversationInsights, Transcriber

router = APIRouter()


def unredacted(conversation: Conversation) -> Conversation:
    """Clear any `redacted` flag set outside the pipeline.

    A client (or an adapter) claiming the text is already redacted must not be able to make the
    pipeline skip PII redaction, so the API never trusts the flag.
    """
    return conversation.model_copy(update={"redacted": False})


async def transcribe_spooled(
    conversation: Conversation, audio_uri: str, spool: AudioSpool, transcriber: Transcriber
) -> Conversation:
    """Transcribe a spooled recording, keeping the caller's start time and metadata."""
    # resolve() stats and probes the file; keep that blocking I/O off the event loop.
    path = await asyncio.to_thread(spool.resolve, audio_uri)
    transcribed = await transcriber.transcribe(str(path), conversation.id)
    return transcribed.model_copy(
        update={
            "started_at": conversation.started_at,
            "metadata": {**conversation.metadata, **transcribed.metadata},
        }
    )


@router.post("/v1/conversations", response_model=ConversationInsights)
async def analyze_conversation(
    conversation: Conversation,
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
    spool: Annotated[AudioSpool, Depends(get_spool)],
    transcriber: Annotated[Transcriber, Depends(get_transcriber)],
) -> ConversationInsights:
    with api_span("api.conversations") as span:
        span.set_attribute("conversation.id", conversation.id)
        span.set_attribute("channel", conversation.channel.value)
        if conversation.audio_uri is not None and not conversation.turns:
            conversation = await transcribe_spooled(
                conversation, conversation.audio_uri, spool, transcriber
            )
        elif conversation.audio_uri is not None:
            # Even with turns present, an unchecked audio_uri must never travel downstream.
            await asyncio.to_thread(spool.resolve, conversation.audio_uri)
        elif not conversation.turns:
            raise HTTPException(422, "A conversation needs turns or an audio_uri.")
        return await pipeline.analyze(unredacted(conversation))


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(state: State) -> JSONResponse:
    if not state.ready:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return JSONResponse({"status": "ready"})


@router.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
