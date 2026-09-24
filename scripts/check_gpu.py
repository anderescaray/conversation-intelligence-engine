"""Check that PyTorch can use the local CUDA GPU.

Run after `uv sync --group ml`:

    uv run python scripts/check_gpu.py

`torch.cuda.is_available()` alone is not enough on Blackwell GPUs (sm_120): a torch build
without sm_120 kernels still reports CUDA as available and only fails at the first kernel
launch. So the script also checks the compiled architectures and runs real matmuls.
Local diagnostic tool, not an application code path, so it emits no spans or metrics.
"""

import time
from typing import Any

_CORRECTNESS_SIZE = 512
_BENCH_SIZE = 4096
_BENCH_ITERS = 10


def _benchmark(torch: Any, dtype: Any) -> float:
    """Return achieved TFLOPS for a square matmul of `_BENCH_SIZE` in `dtype`."""
    a = torch.randn(_BENCH_SIZE, _BENCH_SIZE, device="cuda", dtype=dtype)
    b = torch.randn(_BENCH_SIZE, _BENCH_SIZE, device="cuda", dtype=dtype)
    torch.matmul(a, b)  # warm-up: first call pays for cuBLAS init and kernel selection
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(_BENCH_ITERS):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / _BENCH_ITERS
    return 2 * _BENCH_SIZE**3 / seconds / 1e12


def main() -> int:
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("torch is not installed. Run: uv sync --group ml")
        return 1

    print(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("CUDA is not available. Check the NVIDIA driver (and WSL GPU support): nvidia-smi")
        return 1

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    arch = f"sm_{props.major}{props.minor}"
    print(f"GPU:  {props.name} ({arch})")
    print(f"VRAM: {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free")

    compiled = torch.cuda.get_arch_list()
    if arch not in compiled:
        print(f"This torch build has no kernels for {arch} (has: {', '.join(compiled)}).")
        return 1

    x = torch.randn(_CORRECTNESS_SIZE, _CORRECTNESS_SIZE)
    y = torch.randn(_CORRECTNESS_SIZE, _CORRECTNESS_SIZE)
    gpu_result = (x.cuda() @ y.cuda()).cpu()
    if not torch.allclose(gpu_result, x @ y, rtol=1e-3, atol=1e-3):
        print("GPU matmul result does not match the CPU result.")
        return 1
    print("Matmul correctness: GPU matches CPU")

    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        print(f"Matmul {_BENCH_SIZE}x{_BENCH_SIZE} {name}: {_benchmark(torch, dtype):.1f} TFLOPS")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
