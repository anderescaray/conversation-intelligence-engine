"""Prometheus metrics for LLM calls, named per `docs/ARCHITECTURE.md#telemetry`."""

from prometheus_client import Counter

LLM_TOKENS = Counter(
    "convintel_llm_tokens",
    "LLM tokens consumed, by model and direction.",
    ["model", "direction"],
)
"""Exposed as `convintel_llm_tokens_total`; `direction` is `in` or `out`."""

LLM_COST_USD = Counter(
    "convintel_llm_cost_usd",
    "LLM spend in USD, by model and calling operator.",
    ["model", "operator"],
)
"""Exposed as `convintel_llm_cost_usd_total`."""
