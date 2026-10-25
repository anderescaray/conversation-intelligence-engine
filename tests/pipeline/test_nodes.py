"""run_operator: fail open on expected errors, loud on bugs, every error counted once."""

import pytest
from pipeline_fakes import make_conversation
from prometheus_client import REGISTRY

from convintel.contracts.schemas import (
    Conversation,
    IntentOutput,
    OperatorResult,
)
from convintel.errors import OperatorError
from convintel.pipeline.nodes import run_operator


class _FakeIntent:
    name = "intent"
    version = "0.0.0-test"

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    async def run(self, conversation: Conversation) -> OperatorResult[IntentOutput]:
        if self._error is not None:
            raise self._error
        return OperatorResult[IntentOutput](
            operator=self.name,
            operator_version=self.version,
            output=IntentOutput(label="card_arrival", confidence=0.95, source="classifier"),
            latency_ms=1.0,
        )


def _errors_counted(error_type: str) -> float:
    value = REGISTRY.get_sample_value(
        "convintel_operator_errors_total", {"operator": "intent", "error_type": error_type}
    )
    return value or 0.0


async def test_returns_the_result_and_no_errors() -> None:
    result, errors = await run_operator(_FakeIntent(), make_conversation())
    assert result is not None
    assert result.output.label == "card_arrival"
    assert errors == []


async def test_expected_failure_fails_open_with_the_type_only() -> None:
    error = OperatorError("intent", "model unavailable")
    result, errors = await run_operator(_FakeIntent(error), make_conversation())
    assert result is None
    assert errors == ["intent:OperatorError"]
    assert "model unavailable" not in "".join(errors)


async def test_expected_failure_is_counted_exactly_once() -> None:
    before = _errors_counted("OperatorError")
    await run_operator(_FakeIntent(OperatorError("intent", "boom")), make_conversation())
    assert _errors_counted("OperatorError") - before == 1


async def test_bugs_propagate() -> None:
    with pytest.raises(TypeError):
        await run_operator(_FakeIntent(TypeError("bug")), make_conversation())
