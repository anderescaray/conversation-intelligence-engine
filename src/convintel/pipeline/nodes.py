"""Operator execution for pipeline nodes.

Every operator call goes through ``run_operator``, so each one gets the same span
and metrics (``traced``) and the same failure policy:

- expected failures (``ConvIntelError``) fail open: no result, and
  ``"<operator>:<ErrorType>"`` in ``errors`` (the type only, never the message);
- anything else is a bug and propagates, so it is loud instead of a silent
  partial result.
"""

from pydantic import BaseModel

from convintel.contracts.operators import Operator
from convintel.contracts.schemas import Conversation, OperatorResult
from convintel.errors import ConvIntelError
from convintel.telemetry import traced


async def run_operator[T: BaseModel](
    operator: Operator[T], conversation: Conversation
) -> tuple[OperatorResult[T] | None, list[str]]:
    try:
        with traced(f"operator.{operator.name}", version=operator.version):
            return await operator.run(conversation), []
    except ConvIntelError as exc:
        return None, [f"{operator.name}:{type(exc).__name__}"]
