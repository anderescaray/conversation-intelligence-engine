"""Price and sampling-capability tables for LLM cost accounting and model construction.

Lives here rather than in `Settings` because prices are reviewed code, not per-deployment
config: a wrong price silently skews the cost SLO, so changes should go through a PR.
Prices are USD per million tokens, list price for the provider's first-party API.
"""

import logging

from pydantic import BaseModel, ConfigDict, Field

from convintel.config import LLMProvider

logger = logging.getLogger(__name__)

ANY_MODEL = "*"
"""Wildcard model for providers billed the same regardless of model (e.g. local Ollama)."""


class ModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


PRICES: dict[tuple[LLMProvider, str], ModelPrice] = {
    ("anthropic", "claude-haiku-4-5"): ModelPrice(input_per_mtok=1.00, output_per_mtok=5.00),
    ("anthropic", "claude-sonnet-5-5"): ModelPrice(input_per_mtok=2.00, output_per_mtok=10.00),
    ("ollama", ANY_MODEL): ModelPrice(input_per_mtok=0.0, output_per_mtok=0.0),
}

REJECTS_SAMPLING_PARAMETERS: frozenset[tuple[LLMProvider, str]] = frozenset(
    {("anthropic", "claude-sonnet-5-5")}
)
"""Models whose API refuses non-default sampling parameters; they run at the provider default."""


def effective_temperature(provider: LLMProvider, model: str, temperature: float) -> float | None:
    """The temperature actually sent, or None when the provider default applies."""
    if (provider, model) in REJECTS_SAMPLING_PARAMETERS:
        return None
    return temperature


def compute_cost(
    provider: LLMProvider, model: str, input_tokens: int, output_tokens: int
) -> float | None:
    """Cost in USD, or None when the model is not priced (cost unknown, not zero)."""
    price = PRICES.get((provider, model)) or PRICES.get((provider, ANY_MODEL))
    if price is None:
        logger.warning("no price for %s:%s; cost_usd left empty", provider, model)
        return None
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000
