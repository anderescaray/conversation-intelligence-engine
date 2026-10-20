"""The pipeline state lets parallel branches report errors in the same step."""

from datetime import UTC, datetime
from typing import get_type_hints

from langgraph.graph import END, START, StateGraph

from convintel.contracts.schemas import Channel, Conversation, Speaker, Turn
from convintel.pipeline.state import PipelineState, PipelineUpdate


def _conversation() -> Conversation:
    return Conversation(
        id="conv-1",
        channel=Channel.CHAT,
        started_at=datetime(2026, 10, 20, tzinfo=UTC),
        turns=(Turn(index=0, speaker=Speaker.CUSTOMER, text="My card <CREDIT_CARD> was declined"),),
        redacted=True,
    )


def test_parallel_branches_can_append_errors_in_the_same_step() -> None:
    def failing_sentiment(state: PipelineState) -> PipelineUpdate:
        return {"sentiment": None, "errors": ["sentiment:TimeoutError"]}

    def failing_summary(state: PipelineState) -> PipelineUpdate:
        return {"summary": None, "errors": ["summary:LLMError"]}

    graph = StateGraph(PipelineState)
    graph.add_node("sentiment", failing_sentiment)
    graph.add_node("summary", failing_summary)
    graph.add_edge(START, "sentiment")
    graph.add_edge(START, "summary")
    graph.add_edge("sentiment", END)
    graph.add_edge("summary", END)

    final = graph.compile().invoke(
        {
            "redacted_conversation": _conversation(),
            "sentiment": None,
            "intent": None,
            "summary": None,
            "errors": [],
        }
    )

    # The order of parallel writes is not part of the contract, so compare sorted.
    assert sorted(final["errors"]) == ["sentiment:TimeoutError", "summary:LLMError"]


def test_update_keys_are_state_keys() -> None:
    # A typo in PipelineUpdate would let nodes write a key the graph does not have.
    assert set(get_type_hints(PipelineUpdate)) <= set(get_type_hints(PipelineState))
