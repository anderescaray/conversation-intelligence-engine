import re
import traceback

import pytest
from pipeline_fakes import (
    TAU,
    FakeIntentClassifier,
    FakeOperator,
    FakeRedactor,
    FakeVerifier,
    intent_output,
    make_conversation,
)
from prometheus_client import REGISTRY

import convintel
from convintel.contracts import Channel
from convintel.contracts.schemas import SentimentOutput, SummaryOutput
from convintel.errors import OperatorError, PipelineError
from convintel.pipeline.pipeline import Pipeline

SECRET = "marisol.q@example.com"
# The pipeline does not look inside these outputs, so unvalidated placeholders are enough.
SENTIMENT = SentimentOutput.model_construct()
SUMMARY = SummaryOutput.model_construct()


class _Parts:
    def __init__(
        self,
        *,
        confidence: float = 0.95,
        bundle_threshold: float | None = TAU,
        redactor: FakeRedactor | None = None,
        sentiment_error: OperatorError | None = None,
        summary_cost: float | None = None,
        summary_tokens: int | None = None,
    ) -> None:
        self.redactor = redactor or FakeRedactor()
        self.sentiment = FakeOperator("sentiment", SENTIMENT, error=sentiment_error)
        self.summary = FakeOperator(
            "summary", SUMMARY, cost_usd=summary_cost, input_tokens=summary_tokens
        )
        self.intent = FakeIntentClassifier(
            intent_output(confidence=confidence), threshold=bundle_threshold
        )
        self.verifier = FakeVerifier()

    def pipeline(self, *, threshold_override: float | None = None) -> Pipeline:
        return Pipeline(
            redactor=self.redactor,
            sentiment=self.sentiment,
            summary=self.summary,
            intent=self.intent,
            verify=self.verifier,
            threshold_override=threshold_override,
        )

    async def ready(self, *, threshold_override: float | None = None) -> Pipeline:
        pipeline = self.pipeline(threshold_override=threshold_override)
        await pipeline.warmup()
        return pipeline


# --- redaction: always, fail closed ---------------------------------------------------------


async def test_redacts_even_when_the_input_claims_to_be_redacted() -> None:
    parts = _Parts()
    insights = await (await parts.ready()).analyze(
        make_conversation(f"mail {SECRET}", redacted=True)
    )
    assert parts.redactor.calls == 1
    assert insights.pii_entity_count == 1


async def test_operators_only_ever_see_the_redacted_conversation() -> None:
    parts = _Parts()
    await (await parts.ready()).analyze(make_conversation(f"mail {SECRET}", redacted=False))
    for operator in (parts.sentiment, parts.summary, parts.intent):
        (seen,) = operator.seen
        assert seen.redacted
        assert SECRET not in repr(seen)


async def test_a_redaction_failure_stops_everything_without_its_message() -> None:
    parts = _Parts(redactor=FakeRedactor(error=OperatorError("pii", f"cannot parse {SECRET}")))
    pipeline = await parts.ready()

    with pytest.raises(PipelineError) as excinfo:
        await pipeline.analyze(make_conversation(f"mail {SECRET}"))

    error = excinfo.value
    assert error.stage == "pii"
    assert error.__cause__ is None
    assert error.__suppress_context__
    assert SECRET not in "".join(traceback.format_exception(error))
    assert parts.sentiment.calls == parts.summary.calls == parts.intent.calls == 0


async def test_a_redactor_that_forgets_the_flag_stops_everything() -> None:
    parts = _Parts(redactor=FakeRedactor(sets_flag=False))
    with pytest.raises(PipelineError):
        await (await parts.ready()).analyze(make_conversation())
    assert parts.sentiment.calls == 0


async def test_a_redactor_bug_propagates_as_a_bug() -> None:
    parts = _Parts(redactor=FakeRedactor(error=TypeError("bug")))
    with pytest.raises(TypeError):
        await (await parts.ready()).analyze(make_conversation())


# --- insights -----------------------------------------------------------------------------


async def test_insights_aggregate_the_branches() -> None:
    parts = _Parts(summary_cost=0.002, summary_tokens=120)
    insights = await (await parts.ready()).analyze(make_conversation(conversation_id="c-7"))

    assert insights.conversation_id == "c-7"
    assert insights.pipeline_version == convintel.__version__
    assert insights.sentiment is not None
    assert insights.summary is not None
    assert insights.intent is not None
    assert insights.total_cost_usd == pytest.approx(0.002)
    assert insights.total_latency_ms > 0
    assert insights.errors == []  # local models (no tokens) are not reported as unknown cost


async def test_an_unknown_llm_cost_is_reported_not_hidden() -> None:
    parts = _Parts(summary_cost=None, summary_tokens=120)
    insights = await (await parts.ready()).analyze(make_conversation())
    assert insights.total_cost_usd == 0
    assert insights.errors == ["summary:UnknownCost"]


async def test_errors_carry_only_the_operator_and_the_error_type() -> None:
    parts = _Parts(
        sentiment_error=OperatorError("sentiment", f"model choked on {SECRET}"),
        summary_tokens=10,
    )
    insights = await (await parts.ready()).analyze(make_conversation())

    assert insights.sentiment is None
    assert len(insights.errors) == 2
    assert all(re.fullmatch(r"[a-z_]+:[A-Za-z]+", e) for e in insights.errors)
    assert SECRET not in repr(insights)


def _latency_count(channel: str) -> float:
    sample = REGISTRY.get_sample_value(
        "convintel_pipeline_latency_seconds_count", {"channel": channel}
    )
    return sample or 0.0


async def test_latency_is_recorded_even_when_the_analysis_fails() -> None:
    before = _latency_count(Channel.CHAT.value)
    parts = _Parts(redactor=FakeRedactor(error=OperatorError("pii", "x")))
    with pytest.raises(PipelineError):
        await (await parts.ready()).analyze(make_conversation())
    assert _latency_count(Channel.CHAT.value) - before == 1


# --- warmup and τ -------------------------------------------------------------------------


async def test_the_bundle_threshold_is_used_without_an_override() -> None:
    parts = _Parts(confidence=0.7)  # below the bundle's τ
    await (await parts.ready()).analyze(make_conversation())
    assert parts.verifier.calls == 1


async def test_the_settings_threshold_overrides_the_bundle() -> None:
    parts = _Parts(confidence=0.7)
    await (await parts.ready(threshold_override=0.5)).analyze(make_conversation())
    assert parts.verifier.calls == 0


async def test_an_override_of_zero_means_never_verify() -> None:
    parts = _Parts(confidence=0.1)
    await (await parts.ready(threshold_override=0.0)).analyze(make_conversation())
    assert parts.verifier.calls == 0


async def test_warmup_fails_without_any_threshold() -> None:
    with pytest.raises(PipelineError):
        await _Parts(bundle_threshold=None).ready()


async def test_analyze_before_warmup_is_a_bug() -> None:
    with pytest.raises(RuntimeError):
        await _Parts().pipeline().analyze(make_conversation())
