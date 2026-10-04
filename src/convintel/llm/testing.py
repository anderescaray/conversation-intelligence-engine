"""Offline fakes for code that calls `LLMClient`, so unit tests never hit a provider.

Shipped in the package (not under `tests/`) so other lanes can reuse them:

    model = FakeStructuredChatModel(responses=[tool_answer(SummaryOutput, {...})])
    client = LLMClient(settings, chat_model=model, sleep=no_sleep)
"""

from collections.abc import Sequence
from typing import Annotated, Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field, SkipValidation


class FakeStructuredChatModel(BaseChatModel):
    """Replays scripted answers in order; an exception in the script is raised instead.

    Supports `with_structured_output` the way real providers do (tool calling): `bind_tools`
    is a no-op and each scripted `AIMessage` carries the tool call. Every received message
    list is kept in `calls` for assertions.
    """

    # Skipped: pydantic would try to coerce scripted exceptions into AIMessage.
    responses: Annotated[list[AIMessage | Exception], SkipValidation]
    calls: list[list[BaseMessage]] = Field(default_factory=list[list[BaseMessage]])

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeStructuredChatModel ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeStructuredChatModel":
        return self


def tool_answer(
    schema: type[BaseModel],
    args: dict[str, Any],
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> AIMessage:
    """An assistant message that answers `schema` through a tool call, with token usage."""
    return AIMessage(
        content="",
        tool_calls=[{"name": schema.__name__, "args": args, "id": "call_0", "type": "tool_call"}],
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


async def no_sleep(_: float) -> None:
    """Drop-in for `asyncio.sleep` so retry tests do not wait."""
