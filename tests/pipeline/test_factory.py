from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import SecretStr

from convintel.config import Settings, get_settings
from convintel.llm import client as client_module
from convintel.operators.intent import operator as intent_module
from convintel.operators.llm import SummaryOperator
from convintel.operators.pii import redactor as redactor_module
from convintel.operators.sentiment import operator as sentiment_module
from convintel.pipeline.factory import build_pipeline


def _fail(*args: Any) -> Any:
    raise AssertionError("build_pipeline() must not load models or read process settings")


@pytest.fixture(autouse=True)
def _no_io(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Any model load, or a fallback to the process-wide settings, fails the test."""
    monkeypatch.setattr(redactor_module, "get_analyzer", _fail)
    monkeypatch.setattr(sentiment_module, "get_sentiment_model", _fail)
    monkeypatch.setattr(intent_module, "get_intent_model", _fail)
    monkeypatch.setattr(client_module, "get_settings", _fail)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


def test_building_does_no_io() -> None:
    # The autouse fixture makes every loader fail: building must reach the end without them.
    pipeline = build_pipeline(_settings())

    assert pipeline._graph is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("threshold", [None, 0.0, 0.42])
def test_the_settings_threshold_is_the_override(threshold: float | None) -> None:
    pipeline = build_pipeline(_settings(intent_verifier_threshold=threshold))

    assert pipeline._threshold_override == threshold  # pyright: ignore[reportPrivateUsage]


def test_the_summary_client_uses_the_given_settings() -> None:
    settings = _settings(
        llm_provider="anthropic", llm_model="claude-test", anthropic_api_key=SecretStr("k")
    )

    pipeline = build_pipeline(settings)

    summary = pipeline._summary  # pyright: ignore[reportPrivateUsage]
    assert isinstance(summary, SummaryOperator)
    assert summary._client.settings is settings  # pyright: ignore[reportPrivateUsage]


# Never 300, the setting's default: a factory that ignored the setting would still pass.
@pytest.mark.parametrize("timeout_s", [0.5, 120.0])
def test_the_settings_warmup_timeout_reaches_the_pipeline(timeout_s: float) -> None:
    pipeline = build_pipeline(_settings(pipeline_warmup_timeout_s=timeout_s))

    assert pipeline._warmup_timeout_s == timeout_s  # pyright: ignore[reportPrivateUsage]
