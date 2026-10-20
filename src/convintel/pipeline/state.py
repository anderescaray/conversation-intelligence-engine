"""State shared by the nodes of the conversation pipeline graph.

The graph only ever holds the *redacted* conversation. PII redaction runs in
``Pipeline.analyze()`` before the graph is invoked, because OpenInference's
LangChain instrumentation records each node's input state on its span: an
unredacted conversation in the state would be exported to the trace backend.
"""

import operator
from typing import Annotated, TypedDict

from convintel.contracts.schemas import (
    Conversation,
    IntentOutput,
    OperatorResult,
    SentimentOutput,
    SummaryOutput,
)


class PipelineState(TypedDict):
    """What every node reads: the full state of one conversation's analysis.

    Each operator result has a single writer per step: ``intent`` is written by
    the intent node and may later be replaced by the verifier, never in
    parallel. ``errors`` is appended to by parallel branches, so it needs a
    reducer that concatenates the lists. Entries are ``"<operator>:<ErrorType>"``:
    the error type only, never its message.
    """

    redacted_conversation: Conversation
    sentiment: OperatorResult[SentimentOutput] | None
    intent: OperatorResult[IntentOutput] | None
    summary: OperatorResult[SummaryOutput] | None
    errors: Annotated[list[str], operator.add]


class PipelineUpdate(TypedDict, total=False):
    """What a node returns: only the keys it changes.

    There is no ``redacted_conversation`` key, so no node can replace the
    conversation the graph was started with.
    """

    sentiment: OperatorResult[SentimentOutput] | None
    intent: OperatorResult[IntentOutput] | None
    summary: OperatorResult[SummaryOutput] | None
    errors: list[str]
