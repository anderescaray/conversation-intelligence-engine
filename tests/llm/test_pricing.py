import logging

import pytest

from convintel.llm.pricing import compute_cost


def test_haiku_cost_uses_per_million_prices() -> None:
    cost = compute_cost("anthropic", "claude-haiku-4-5", 1_000_000, 200_000)
    assert cost == pytest.approx(1.00 + 0.2 * 5.00)


def test_ollama_is_free_for_any_model() -> None:
    assert compute_cost("ollama", "llama3.1:8b", 5000, 5000) == 0.0
    assert compute_cost("ollama", "qwen2.5:7b", 1, 1) == 0.0


def test_unknown_model_cost_is_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert compute_cost("anthropic", "claude-unknown", 10, 10) is None
    assert "anthropic:claude-unknown" in caplog.text
