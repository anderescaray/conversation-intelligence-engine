import pickle

import pytest

from convintel.errors import ConvIntelError, LLMError, OperatorError, TranscriptionError


@pytest.mark.parametrize("error_type", [OperatorError, TranscriptionError, LLMError])
def test_domain_errors_share_a_base(error_type: type[Exception]) -> None:
    assert issubclass(error_type, ConvIntelError)


def test_operator_error_keeps_operator_name() -> None:
    error = OperatorError("intent", "model not loaded")
    assert error.operator == "intent"
    assert str(error) == "[intent] model not loaded"


def test_operator_error_survives_pickling() -> None:
    restored = pickle.loads(pickle.dumps(OperatorError("intent", "model not loaded")))
    assert isinstance(restored, OperatorError)
    assert restored.operator == "intent"
    assert str(restored) == "[intent] model not loaded"
