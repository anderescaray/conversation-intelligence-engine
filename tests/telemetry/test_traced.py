import asyncio
import time

import pytest
from prometheus_client import REGISTRY

from convintel.telemetry import traced


def _latency_count(operator: str, version: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "convintel_operator_latency_seconds_count",
            {"operator": operator, "version": version},
        )
        or 0.0
    )


def _error_count(operator: str, error_type: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "convintel_operator_errors_total",
            {"operator": operator, "error_type": error_type},
        )
        or 0.0
    )


def test_sync_decorator_records_latency_and_returns_value() -> None:
    @traced("operator.sentiment", version="1.2.3")
    def run() -> int:
        time.sleep(0.001)
        return 42

    before = _latency_count("sentiment", "1.2.3")
    assert run() == 42
    assert _latency_count("sentiment", "1.2.3") == before + 1


def test_async_decorator_records_latency() -> None:
    @traced("operator.intent", version="2.0.0")
    async def run() -> str:
        await asyncio.sleep(0)
        return "ok"

    before = _latency_count("intent", "2.0.0")
    assert asyncio.run(run()) == "ok"
    assert _latency_count("intent", "2.0.0") == before + 1


def test_context_manager_records_latency() -> None:
    before = _latency_count("pii", "unknown")
    with traced("operator.pii"):
        pass
    assert _latency_count("pii", "unknown") == before + 1


def test_error_is_counted_and_reraised() -> None:
    @traced("operator.summary", version="0.1.0")
    def run() -> None:
        raise ValueError("boom")

    lat_before = _latency_count("summary", "0.1.0")
    err_before = _error_count("summary", "ValueError")
    with pytest.raises(ValueError, match="boom"):
        run()
    assert _error_count("summary", "ValueError") == err_before + 1
    assert _latency_count("summary", "0.1.0") == lat_before + 1


def test_context_manager_error_is_counted() -> None:
    err_before = _error_count("pii", "KeyError")
    with pytest.raises(KeyError), traced("operator.pii"):
        raise KeyError("x")
    assert _error_count("pii", "KeyError") == err_before + 1


def test_method_picks_version_from_instance() -> None:
    class FakeOperator:
        version = "3.1.4"

        @traced("operator.intent")
        def run(self) -> int:
            return 1

    before = _latency_count("intent", "3.1.4")
    assert FakeOperator().run() == 1
    assert _latency_count("intent", "3.1.4") == before + 1


def test_decorator_preserves_function_metadata() -> None:
    @traced("operator.sentiment")
    def my_operator() -> None:
        """Docstring."""

    assert my_operator.__name__ == "my_operator"
    assert my_operator.__doc__ == "Docstring."
