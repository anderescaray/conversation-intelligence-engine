"""The production app end to end over HTTP: what uvicorn serves with `--factory`.

`create_production_app()` with the real `Settings`, served by a `TestClient` used as a context
manager, so the lifespan runs the real telemetry setup and the real warmups. Needs
`docker compose up` (MLflow serving `intent-classifier@champion`, Phoenix receiving spans), the
`ml` group and `ANTHROPIC_API_KEY`; the summary calls the configured provider (Haiku by
default) once per run, a fraction of a cent. Every name, number and address below is
fictitious (the same values as `tests/pipeline/test_pipeline_integration.py`).

The lifespan sweeps the default spool (`var/spool/audio`, relative to the working directory)
and deletes files older than 24 h, as production does: never run this test from a checkout
whose `var/spool/audio` holds real audio. Known risk until the spool path is a setting (#27).

The lifespan calls the production `telemetry.setup()`, which installs the process-wide tracer
provider (OTLP to Phoenix); OpenTelemetry allows that only once per process and `setup()` is
idempotent, so whichever module calls it first owns the provider. Run this file alone or under
`-m integration`, never in one process with `tests/telemetry/test_setup.py` (e.g. `-m ""`):
if this test ran first, `test_setup.py` would not see its own spans.
"""

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from convintel.api.main import create_production_app
from convintel.config import get_settings
from convintel.contracts import Channel, Conversation, ConversationInsights, Speaker, Turn

pytestmark = pytest.mark.integration

C, A = Speaker.CUSTOMER, Speaker.AGENT

NAME = "Priya Thistlewood"
CARD = "4111 1111 1111 1111"  # published issuer test number, Luhn-valid
EMAIL = "priya.t@example.com"
PHONE = "415-555-0123"  # 555-01xx is reserved for fiction
PLANTED_VALUES = (
    NAME,
    *NAME.split(),
    CARD,
    CARD.replace(" ", ""),
    EMAIL,
    PHONE,
    PHONE.replace("-", ""),
)

CONVERSATION = Conversation(
    id="it-api-1",
    channel=Channel.CHAT,
    started_at=datetime(2026, 10, 20, tzinfo=UTC),
    turns=tuple(
        Turn(index=i, speaker=speaker, text=text)
        for i, (speaker, text) in enumerate(
            (
                (C, f"Hi, this is {NAME}."),
                (A, "Good morning, how can I help you today?"),
                (C, "I think someone stole my card, I can't find it anywhere."),
                (A, f"I'm sorry to hear that. I'll block card {CARD} now. Is {EMAIL} correct?"),
                (C, f"Yes, and you can call me on {PHONE}."),
                (A, "Done. A replacement card is on its way."),
            )
        )
    ),
)


@pytest.fixture(autouse=True)
def no_network() -> None:
    """Overrides the conftest's socket block: this test needs MLflow, Phoenix and the LLM."""


@pytest.fixture
def client() -> Iterator[TestClient]:
    for module in ("torch", "transformers", "mlflow", "en_core_web_lg"):
        pytest.importorskip(module)
    # The real Settings (env and .env), read fresh rather than from an earlier test's cache.
    get_settings.cache_clear()
    try:
        with TestClient(create_production_app()) as test_client:
            yield test_client
    finally:
        get_settings.cache_clear()


def test_the_production_app_analyzes_a_conversation_over_http(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert client.get("/readyz").status_code == 200

    response = client.post("/v1/conversations", json=CONVERSATION.model_dump(mode="json"))

    assert response.status_code == 200, response.text
    leaked = sorted(value for value in PLANTED_VALUES if value in response.text)
    assert leaked == [], f"planted values in the response body: {leaked}"
    insights = ConversationInsights.model_validate_json(response.text)
    with capsys.disabled():
        print(
            f"\nAPI integration: cost ${insights.total_cost_usd:.6f}, "
            f"analyze {insights.total_latency_ms:.0f} ms"
        )
    assert insights.conversation_id == CONVERSATION.id
    assert insights.errors == []
    assert insights.sentiment is not None
    assert insights.intent is not None
    assert insights.summary is not None
    assert insights.pii_entity_count > 0
