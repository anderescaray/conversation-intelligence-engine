from prometheus_client import REGISTRY, Counter, Histogram

from convintel.telemetry import metrics

# metric name -> (module constant, labels for a probe series, sample suffix to look up)
_EXPECTED: dict[str, tuple[object, dict[str, str], str]] = {
    "convintel_requests_total": (
        metrics.REQUESTS_TOTAL,
        {"route": "/probe", "status": "200"},
        "",
    ),
    "convintel_pipeline_latency_seconds": (
        metrics.PIPELINE_LATENCY_SECONDS,
        {"channel": "probe"},
        "_count",
    ),
    "convintel_operator_latency_seconds": (
        metrics.OPERATOR_LATENCY_SECONDS,
        {"operator": "probe", "version": "0"},
        "_count",
    ),
    "convintel_operator_errors_total": (
        metrics.OPERATOR_ERRORS_TOTAL,
        {"operator": "probe", "error_type": "ProbeError"},
        "",
    ),
    "convintel_intent_verifier_invocations_total": (
        metrics.INTENT_VERIFIER_INVOCATIONS_TOTAL,
        {"outcome": "agreed"},
        "",
    ),
    "convintel_asr_realtime_factor": (
        metrics.ASR_REALTIME_FACTOR,
        {"model": "probe"},
        "_count",
    ),
}


def test_metrics_expose_the_architecture_names_and_labels() -> None:
    for name, (metric, labels, suffix) in _EXPECTED.items():
        assert isinstance(metric, Counter | Histogram)
        if isinstance(metric, Counter):
            metric.labels(**labels).inc()
        else:
            metric.labels(**labels).observe(0.1)
        assert REGISTRY.get_sample_value(name + suffix, labels) is not None


def test_no_extra_pipeline_metrics_are_declared() -> None:
    declared = {m for m in vars(metrics).values() if isinstance(m, Counter | Histogram)}
    assert len(declared) == len(_EXPECTED)


def test_llm_metrics_are_not_declared_here() -> None:
    # Owned by convintel.llm.metrics (T04); declaring them here would collide at import.
    assert not any("llm" in name.lower() for name in vars(metrics))


def test_llm_metrics_module_coexists_without_duplicate_timeseries() -> None:
    """Both metric modules share the default REGISTRY; importing them in the same
    process must not raise prometheus_client's "Duplicated timeseries" error."""
    import convintel.llm.metrics as llm_metrics

    llm_metrics.LLM_TOKENS.labels(model="probe", direction="in").inc()
    metrics.REQUESTS_TOTAL.labels(route="/probe", status="200").inc()

    assert (
        REGISTRY.get_sample_value(
            "convintel_llm_tokens_total", {"model": "probe", "direction": "in"}
        )
        is not None
    )
    assert (
        REGISTRY.get_sample_value("convintel_requests_total", {"route": "/probe", "status": "200"})
        is not None
    )
