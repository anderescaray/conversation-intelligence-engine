"""LLM summary and structured extraction of a conversation (`SummaryOutput`)."""

import time

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from convintel.contracts import Conversation, OperatorResult, SummaryOutput
from convintel.errors import LLMError, OperatorError
from convintel.llm import LLMClient, Prompt, build_chat_model, load_prompt, wrap_conversation


class SummaryOperator:
    """Implements `Operator[SummaryOutput]` with one structured LLM call.

    Expects a redacted conversation: the text is sent to the configured provider as is.
    Call `warmup()` at startup so a bad LLM configuration fails startup, not the first request.
    """

    name = "summary"
    version = "1.0.0"

    def __init__(self, client: LLMClient | None = None, prompt: Prompt | None = None) -> None:
        self._client = client or LLMClient()
        self._prompt = prompt or load_prompt("summary", 1)

    async def warmup(self) -> None:
        """Validate the LLM configuration (provider wired up, API key present) without network.

        Builds the chat model the way the client will, with ``build_chat_model()``, and throws
        it away: construction does no I/O, and an LLM call per startup would cost money and
        tie startup to the provider's availability.
        """
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            f"operator.{self.name}.warmup", record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("operator.version", self.version)
            span.set_attribute("prompt.version", self._prompt.id)
            span.set_attribute("model.id", self._client.model_id)
            try:
                try:
                    build_chat_model(self._client.settings, temperature=self._prompt.temperature)
                except LLMError as exc:
                    # Built from settings names, never from conversation text: safe to surface.
                    raise OperatorError(self.name, str(exc)) from None
            except Exception as exc:
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def run(self, conversation: Conversation) -> OperatorResult[SummaryOutput]:
        start = time.perf_counter()
        # The caller opens the span with traced() and records latency and errors once; the
        # operator only annotates the current span (a no-op when there is none).
        span = trace.get_current_span()
        span.set_attribute("conversation.id", conversation.id)
        span.set_attribute("prompt.version", self._prompt.id)
        span.set_attribute("model.id", self._client.model_id)
        try:
            response = await self._client.structured(
                SummaryOutput,
                self._prompt,
                operator=self.name,
                channel=conversation.channel.value,
                language=conversation.language,
                conversation=wrap_conversation(conversation.text),
            )
        except LLMError as exc:
            raise OperatorError(self.name, str(exc)) from exc
        span.set_attribute("llm.tokens.input", response.input_tokens)
        span.set_attribute("llm.tokens.output", response.output_tokens)
        if response.cost_usd is not None:
            span.set_attribute("llm.cost_usd", response.cost_usd)

        return OperatorResult[SummaryOutput](
            operator=self.name,
            operator_version=self.version,
            output=response.output,
            latency_ms=(time.perf_counter() - start) * 1000,
            model_id=response.model_id,
            prompt_version=self._prompt.id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )
