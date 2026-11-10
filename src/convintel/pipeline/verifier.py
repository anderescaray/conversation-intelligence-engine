"""LLM intent verifier: a second opinion on a low-confidence classifier intent.

The LLM may only choose among the classifier's top-k candidates, or reject them all. It sees each
candidate as a letter and its pinned taxonomy description, never the label name, the
classifier's probabilities or which candidate the classifier preferred, and the candidates come in
a per-conversation shuffled order: nothing in the prompt anchors it on the classifier's answer.
It reads the whole redacted conversation; only the customer's turns state the reason.

Outcomes (``convintel_intent_verifier_invocations_total{outcome}``):

- ``confirmed``: the classifier's top-1 label;
- ``overridden``: another candidate, with the classifier's calibrated probability for it;
- ``rejected``: none of them fits, so the intent becomes ``unknown`` with confidence 0;
- ``failed``: an expected LLM failure; the graph keeps the classifier's result.

The model's ``reasoning`` exists only to make it think before choosing. It is dropped as soon as
the answer is parsed: never in spans, logs, errors or the insights.
"""

import hashlib
import random
import time
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any, Literal

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field, create_model

from convintel.contracts import Conversation, IntentOutput, OperatorResult
from convintel.errors import ConvIntelError, LLMError, OperatorError
from convintel.llm import LLMClient, Prompt, build_chat_model, load_prompt, wrap_conversation
from convintel.operators.intent.descriptions import intent_descriptions
from convintel.operators.intent.operator import UNKNOWN_LABEL
from convintel.telemetry import INTENT_VERIFIER_INVOCATIONS_TOTAL

LETTERS = ("A", "B", "C", "D", "E")
NONE_CHOICE = "NONE"
VERIFIER_SOURCE = "llm_verifier"
DESCRIPTIONS_VERSION = 3
PROMPT_NAME = "intent_verifier"
PROMPT_VERSION = 1

type Outcome = Literal["confirmed", "overridden", "rejected", "failed"]


class VerifierAnswer(BaseModel):
    """Common shape of every answer model; ``answer_model(n)`` narrows ``choice``."""

    reasoning: str = Field(description="One or two short sentences explaining the choice.")
    choice: str


@cache
def answer_model(n: int) -> type[VerifierAnswer]:
    """The answer schema for ``n`` candidates: ``reasoning`` first, then one of the first ``n``
    letters or NONE.

    One model per candidate count, so a letter with no candidate behind it fails validation and
    gets the client's single repair. ``reasoning`` comes first so the model reasons before it
    commits to a letter. At most five classes ever exist.
    """
    if not 1 <= n <= len(LETTERS):
        raise ValueError(f"the verifier takes 1 to {len(LETTERS)} candidates, got {n}")
    choice: Any = Literal[(*LETTERS[:n], NONE_CHOICE)]
    return create_model(
        "IntentVerdict",
        __base__=VerifierAnswer,
        __doc__="The candidate letter that matches the customer's reason for contacting.",
        choice=(choice, Field(description="Exactly one candidate letter, or NONE.")),
    )


def shuffled_candidates(conversation_id: str, labels: Sequence[str]) -> list[str]:
    """The candidates in an order fixed by the conversation id and unrelated to their ranking.

    Seeded with sha256, never Python's ``hash()``, which is salted per process: the same
    conversation must show the same prompt across workers and reruns.
    """
    if not 1 <= len(labels) <= len(LETTERS):
        raise ValueError(f"the verifier takes 1 to {len(LETTERS)} candidates, got {len(labels)}")
    seed = int.from_bytes(hashlib.sha256(conversation_id.encode("utf-8")).digest()[:8], "big")
    order = list(labels)
    random.Random(seed).shuffle(order)
    return order


def render_candidates(labels: Sequence[str], descriptions: Mapping[str, str]) -> str:
    """One line per candidate, ``"A. <description>"``: the description only, never the label."""
    return "\n".join(
        f"{letter}. {descriptions[label]}" for letter, label in zip(LETTERS, labels, strict=False)
    )


def label_for(choice: str, labels: Sequence[str]) -> str | None:
    """The label behind a letter, or None for NONE. A letter past the candidates is a bug."""
    if choice == NONE_CHOICE:
        return None
    index = LETTERS.index(choice)
    if index >= len(labels):
        raise ValueError(f"choice {choice!r} has no candidate behind it")
    return labels[index]


def verified_output(classified: IntentOutput, chosen: str | None) -> tuple[Outcome, IntentOutput]:
    """Turn the verifier's choice into the intent the pipeline returns.

    The confidence stays the classifier's calibrated probability for the chosen label: the LLM
    gives no calibrated number of its own. ``top_k`` is kept in every case for auditing.
    """
    if chosen is None:
        output = IntentOutput(
            label=UNKNOWN_LABEL, confidence=0.0, source=VERIFIER_SOURCE, top_k=classified.top_k
        )
        return "rejected", output
    outcome: Outcome = "confirmed" if chosen == classified.label else "overridden"
    probabilities = dict(classified.top_k)
    output = IntentOutput(
        label=chosen,
        confidence=probabilities[chosen],
        source=VERIFIER_SOURCE,
        top_k=classified.top_k,
    )
    return outcome, output


class LLMIntentVerifier:
    """Verifies a low-confidence intent with one structured LLM call.

    The constructor does no I/O; ``warmup()`` loads the prompt and validates the LLM
    configuration, so a bad configuration fails startup, not the first low-confidence request.
    Expects a redacted conversation: its text is sent to the configured provider as is.
    """

    name = "intent_verifier"
    version = "1.0.0"

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self._descriptions = intent_descriptions(DESCRIPTIONS_VERSION)
        self._prompt: Prompt | None = None

    async def warmup(self) -> None:
        """Load the prompt and validate the LLM configuration without any network call.

        Same as the summary operator: ``build_chat_model()`` does no I/O, and an LLM call per
        startup would cost money and tie startup to the provider's availability.
        """
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            f"operator.{self.name}.warmup", record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("operator.version", self.version)
            span.set_attribute("model.id", self._client.model_id)
            try:
                try:
                    prompt = load_prompt(PROMPT_NAME, PROMPT_VERSION)
                    build_chat_model(self._client.settings, temperature=prompt.temperature)
                except LLMError as exc:
                    # Built from file and settings names, never from conversation text.
                    raise OperatorError(self.name, str(exc)) from None
            except Exception as exc:
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise
            span.set_attribute("prompt.version", prompt.id)
            self._prompt = prompt

    async def __call__(
        self, conversation: Conversation, classified: OperatorResult[IntentOutput]
    ) -> tuple[OperatorResult[IntentOutput] | None, list[str]]:
        """The verified intent, or ``(None, ["intent_verifier:<ErrorType>"])`` on an expected
        failure, so the graph keeps the classifier's result. Anything else is a bug and
        propagates."""
        if self._prompt is None:
            raise RuntimeError("LLMIntentVerifier called before warmup()")
        prompt = self._prompt
        start = time.perf_counter()
        labels = shuffled_candidates(
            conversation.id, [label for label, _ in classified.output.top_k]
        )
        # The caller opens the span with traced(); the verifier only annotates it.
        span = trace.get_current_span()
        span.set_attribute("conversation.id", conversation.id)
        span.set_attribute("verifier.candidates", len(labels))
        span.set_attribute("prompt.version", prompt.id)
        span.set_attribute("model.id", self._client.model_id)
        try:
            response = await self._client.structured(
                answer_model(len(labels)),
                prompt,
                operator=self.name,
                channel=conversation.channel.value,
                language=conversation.language,
                candidates=render_candidates(labels, self._descriptions),
                conversation=wrap_conversation(conversation.text),
            )
        except ConvIntelError as exc:
            # The type only: the message is not needed, and `reasoning` never leaves this call.
            error_type = type(exc).__name__
            span.set_attribute("verifier.outcome", "failed")
            span.set_attribute("error.type", error_type)
            span.set_status(Status(StatusCode.ERROR))
            INTENT_VERIFIER_INVOCATIONS_TOTAL.labels(outcome="failed").inc()
            return None, [f"{self.name}:{error_type}"]

        # Only the letter survives; the reasoning is dropped here.
        chosen = label_for(response.output.choice, labels)
        outcome, output = verified_output(classified.output, chosen)
        span.set_attribute("verifier.outcome", outcome)
        span.set_attribute("llm.tokens.input", response.input_tokens)
        span.set_attribute("llm.tokens.output", response.output_tokens)
        if response.cost_usd is not None:
            span.set_attribute("llm.cost_usd", response.cost_usd)
        INTENT_VERIFIER_INVOCATIONS_TOTAL.labels(outcome=outcome).inc()

        # It fills the intent slot: `source` records who decided, the rest is the verifier's call.
        return OperatorResult[IntentOutput](
            operator=classified.operator,
            operator_version=classified.operator_version,
            output=output,
            latency_ms=classified.latency_ms + (time.perf_counter() - start) * 1000,
            model_id=response.model_id,
            prompt_version=prompt.id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        ), []
