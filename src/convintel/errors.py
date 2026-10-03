"""Domain exceptions.

A single base class lets the API and the pipeline catch every expected failure in one place
while letting genuine bugs (TypeError, KeyError...) propagate.
"""


class ConvIntelError(Exception):
    """Base class for every expected failure in convintel."""


class OperatorError(ConvIntelError):
    """An operator failed to analyze a conversation.

    Carries the operator name because it is the `operator` label of
    `convintel_operator_errors_total` and the pipeline must record it without parsing messages.
    Both constructor arguments go to `args` so the error survives pickling (process pools,
    task queues), which rebuilds exceptions as `cls(*args)`.
    """

    def __init__(self, operator: str, message: str) -> None:
        super().__init__(operator, message)
        self.operator = operator
        self.message = message

    def __str__(self) -> str:
        return f"[{self.operator}] {self.message}"


class TranscriptionError(ConvIntelError):
    """Audio could not be turned into a Conversation."""


class LLMError(ConvIntelError):
    """An LLM call failed after retries, or returned output that does not fit the schema."""
