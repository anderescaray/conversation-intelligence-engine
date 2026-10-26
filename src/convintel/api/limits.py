"""Request body caps enforced before any parsing, so a huge body never reaches pydantic."""

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import HTTPException, Request, Response
from fastapi.routing import APIRoute
from starlette.types import Message

# A long chat transcript is tens of KB; 1 MiB leaves headroom without letting one request
# pin the worker's memory or CPU in JSON parsing. Audio travels by spool path, not in the body.
MAX_CONVERSATION_BYTES = 1024 * 1024
BODY_TOO_LARGE = "Request body too large."


async def read_capped_body(request: Request, limit: int) -> bytes:
    """Read the body, failing with 413 as soon as it exceeds `limit` (declared or streamed)."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, BODY_TOO_LARGE)
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise HTTPException(413, BODY_TOO_LARGE)
    return bytes(body)


class CappedBodyRoute(APIRoute):
    """Route that reads its body under `MAX_CONVERSATION_BYTES` before FastAPI parses it.

    FastAPI reads and decodes the body before any dependency runs, so the cap has to wrap the
    route handler; the checked bytes are then replayed to it.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def capped(request: Request) -> Response:
            body = await read_capped_body(request, MAX_CONVERSATION_BYTES)
            replayed = False

            async def replay() -> Message:
                nonlocal replayed
                if replayed:
                    return await request.receive()
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}

            return await handler(Request(request.scope, replay))

        return capped
