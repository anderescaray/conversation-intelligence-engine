"""Runtime configuration, read from the environment (and `.env` in development).

Env var names carry no prefix so that standard names like `ANTHROPIC_API_KEY` work unchanged
with provider SDKs and existing shells. Secrets have no defaults: a missing key surfaces as
`None` at the call site instead of a silently wrong credential.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "ollama", "google", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: LLMProvider = "anthropic"
    llm_model: str = "claude-haiku-4-5"
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    # Observability and registries
    phoenix_collector_endpoint: str = "http://localhost:6006"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Storage
    database_url: str = "postgresql://localhost:5432/convintel"

    # Twilio
    twilio_account_sid: str | None = None
    twilio_auth_token: SecretStr | None = None
    twilio_phone_number: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings, parsed once. Tests call `get_settings.cache_clear()` to reload."""
    return Settings()
