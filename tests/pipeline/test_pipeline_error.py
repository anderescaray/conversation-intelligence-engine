import pickle

from convintel.errors import ConvIntelError, PipelineError


def test_is_an_expected_failure() -> None:
    assert issubclass(PipelineError, ConvIntelError)


def test_carries_the_stage_only() -> None:
    error = PipelineError("pii")
    assert error.stage == "pii"
    assert str(error) == "pipeline stopped at pii"


def test_survives_pickling() -> None:
    restored = pickle.loads(pickle.dumps(PipelineError("pii")))
    assert isinstance(restored, PipelineError)
    assert restored.stage == "pii"
