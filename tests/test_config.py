from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from convintel.config import Settings, get_settings

_ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"

_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "OLLAMA_BASE_URL",
    "INTENT_VERIFIER_THRESHOLD",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the developer's shell and .env from leaking into assertions."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _settings() -> Settings:
    """Build Settings from the environment only, ignoring any local .env file."""
    return Settings(_env_file=None)  # pyright: ignore[reportCallIssue]


def test_defaults_point_to_anthropic_and_local_ollama() -> None:
    settings = _settings()
    assert settings.llm_provider == "anthropic"
    assert settings.llm_model == "claude-haiku-4-5"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.anthropic_api_key is None


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = _settings()
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "llama3.1:8b"
    assert settings.google_api_key is not None
    assert settings.google_api_key.get_secret_value() == "test-key"


def test_empty_values_count_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Copying .env.example verbatim leaves `KEY=` lines; they must not become "" credentials.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("INTENT_VERIFIER_THRESHOLD", "")
    settings = _settings()
    assert settings.anthropic_api_key is None
    assert settings.twilio_account_sid is None
    assert settings.llm_model == "claude-haiku-4-5"
    assert settings.intent_verifier_threshold is None


def test_settings_are_frozen() -> None:
    settings = _settings()
    with pytest.raises(ValidationError):
        settings.llm_model = "other"


def test_env_example_lists_every_setting() -> None:
    keys = {
        line.split("=", 1)[0].strip().lower()
        for line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    assert keys == set(Settings.model_fields)


def test_secrets_are_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    assert "sk-should-not-leak" not in repr(_settings())


def test_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    with pytest.raises(ValidationError):
        _settings()


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_intent_verifier_threshold_defaults_to_none() -> None:
    # None means "use the tau calibrated in the model bundle".
    assert _settings().intent_verifier_threshold is None


@pytest.mark.parametrize("raw", ["0", "0.9", "1"])
def test_reads_intent_verifier_threshold(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    # 0 turns the verifier off and 1 sends (almost) everything to it: both are valid policies.
    monkeypatch.setenv("INTENT_VERIFIER_THRESHOLD", raw)
    assert _settings().intent_verifier_threshold == float(raw)


@pytest.mark.parametrize("raw", ["86", "-0.1", "1.01"])
def test_rejects_out_of_range_intent_verifier_threshold(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    # "86" is the realistic mistake: a percentage instead of a probability.
    monkeypatch.setenv("INTENT_VERIFIER_THRESHOLD", raw)
    with pytest.raises(ValidationError, match="intent_verifier_threshold"):
        _settings()


def test_pipeline_warmup_timeout_defaults_to_300_seconds() -> None:
    # Same value as the old constant in pipeline.py: making it configurable changes no behaviour.
    assert _settings().pipeline_warmup_timeout_s == 300.0


def test_reads_pipeline_warmup_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_WARMUP_TIMEOUT_S", "120")
    assert _settings().pipeline_warmup_timeout_s == 120.0


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf"])
def test_rejects_invalid_pipeline_warmup_timeout(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    # 0 or less would fail every startup; "inf" would silently disable the timeout and is the
    # case only `allow_inf_nan=False` catches (verified by mutation). Pydantic's `gt=0` already
    # rejects "nan"; that case guards the behaviour if the validation changes.
    monkeypatch.setenv("PIPELINE_WARMUP_TIMEOUT_S", raw)
    with pytest.raises(ValidationError, match="pipeline_warmup_timeout_s"):
        _settings()
