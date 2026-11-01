"""The production pipeline end to end: real models, real services, production telemetry.

`build_pipeline(Settings())` -> `warmup()` -> `analyze()` on one synthetic conversation. Needs
`docker compose up` (MLflow serving `intent-classifier@champion`), the `ml` group and
`ANTHROPIC_API_KEY`; the summary calls the configured provider (Haiku by default) once per
module, a fraction of a cent. Every name, number and address below is fictitious.

Spans are captured with `telemetry.setup()` itself (global provider and LangChain
instrumentation, inputs and outputs recorded as in production), exporting to memory.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from convintel.config import Settings
from convintel.contracts import Channel, Conversation, ConversationInsights, Speaker, Turn
from convintel.pipeline.factory import build_pipeline
from convintel.telemetry import setup, tracing

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

C, A = Speaker.CUSTOMER, Speaker.AGENT

NAME = "Priya Thistlewood"
CARD = "4111 1111 1111 1111"  # published issuer test number, Luhn-valid
EMAIL = "priya.t@example.com"
PHONE = "415-555-0123"  # 555-01xx is reserved for fiction
PLANTED_ENTITIES = 4
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
    id="it-pipeline-1",
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


@dataclass(frozen=True)
class Run:
    insights: ConversationInsights
    spans: tuple[ReadableSpan, ...]


def _capture_spans() -> InMemorySpanExporter:
    """Production `setup()`, exporting to memory instead of Phoenix.

    The global tracer provider can be set only once per process. If another test module
    already called `setup()`, its provider is the global one, so the exporter is attached to
    it; otherwise `setup()` installs it.
    """
    exporter = InMemorySpanExporter()
    already_set_up = tracing._provider is not None  # pyright: ignore[reportPrivateUsage]
    provider = setup(settings=Settings(), exporter=exporter)
    if already_set_up:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    assert trace.get_tracer_provider() is provider
    return exporter


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def run() -> Run:
    for module in ("torch", "transformers", "mlflow", "en_core_web_lg"):
        pytest.importorskip(module)
    exporter = _capture_spans()
    pipeline = build_pipeline(Settings())

    await pipeline.warmup()
    insights = await pipeline.analyze(CONVERSATION)

    return Run(insights, exporter.get_finished_spans())


def _texts(span: ReadableSpan) -> Iterator[str]:
    """Everything a span exports that could carry text."""
    yield span.name
    yield from (str(value) for value in (span.attributes or {}).values())
    if span.status.description:
        yield span.status.description
    for event in span.events:
        yield event.name
        yield from (str(value) for value in (event.attributes or {}).values())


def _ms(span: ReadableSpan) -> float:
    assert span.start_time is not None and span.end_time is not None
    return (span.end_time - span.start_time) / 1e6


async def test_the_real_pipeline_analyzes_a_conversation(
    run: Run, capsys: pytest.CaptureFixture[str]
) -> None:
    insights = run.insights

    with capsys.disabled():
        print(f"\nPipeline integration ({insights.total_latency_ms:.0f} ms analyze):")
        for span in sorted(run.spans, key=lambda s: s.start_time or 0):
            if span.name.startswith(("operator.", "pipeline.")):
                print(f"  {span.name:<26} {_ms(span):8.0f} ms")
        langchain = [s for s in run.spans if "openinference.span.kind" in (s.attributes or {})]
        in_trace = sum(
            format(s.context.trace_id, "032x") == insights.trace_id
            for s in langchain
            if s.context is not None
        )
        print(f"  LangChain spans: {len(langchain)} ({in_trace} in the analyze trace)")
        print(f"  cost: ${insights.total_cost_usd:.6f}")

    assert insights.errors == []
    assert insights.sentiment is not None
    assert insights.intent is not None
    assert insights.summary is not None
    assert insights.intent.output.label == "lost_or_stolen_card"
    assert insights.pii_entity_count == PLANTED_ENTITIES
    assert insights.summary.cost_usd is not None
    assert insights.total_cost_usd > 0
    assert insights.total_cost_usd == insights.summary.cost_usd
    assert insights.total_latency_ms > 0


async def test_no_planted_pii_reaches_telemetry(run: Run) -> None:
    exported = [text for span in run.spans for text in _texts(span)]
    assert sorted({value for value in PLANTED_VALUES for text in exported if value in text}) == []
    # Not vacuous: span content is captured, it just holds the redacted text.
    assert any("<EMAIL_ADDRESS>" in text for text in exported)
    (root,) = (span for span in run.spans if span.name == "pipeline.analyze")
    assert root.context is not None
    assert format(root.context.trace_id, "032x") == run.insights.trace_id
