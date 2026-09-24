import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_gpu.py"


def test_check_gpu_passes_on_a_cuda_machine() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    spec = importlib.util.spec_from_file_location("check_gpu", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
