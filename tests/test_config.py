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
    settings = _settings()
    assert settings.anthropic_api_key is None
    assert settings.twilio_account_sid is None
    assert settings.llm_model == "claude-haiku-4-5"


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
