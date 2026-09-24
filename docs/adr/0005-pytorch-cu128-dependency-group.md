# ADR-0005: PyTorch as an opt-in `ml` dependency group on the CUDA 12.8 index

**Status:** Accepted · **Date:** 2026-09

## Context
Fine-tuning (T03) and local model work need PyTorch with GPU support. The development machine
has an RTX 5060 Laptop GPU (Blackwell, `sm_120`). Only torch ≥ 2.7 built for CUDA ≥ 12.8
ships `sm_120` kernels. Older or CPU builds import fine but fail at the first kernel launch.
PyPI's default Linux torch wheels are not guaranteed to match that CUDA build. torch is also
multi-GB, and GitHub runners have no GPU, so installing it in CI would only cost time.

## Decision
- torch lives in a uv **dependency group** `ml` (`uv sync --group ml`), not in `[project]`
  dependencies or an extra. It is a local development tool, so it stays out of the package metadata.
- `[tool.uv.sources]` pins torch to the `pytorch-cu128` index on Linux and Windows. The index is
  `explicit`, so only torch comes from it. macOS falls back to PyPI so `uv lock` still resolves on
  every platform.
- `ml` is not in uv's default groups, and CI runs `uv sync --locked --no-group ml`.
- `scripts/check_gpu.py` checks the setup end to end: CUDA availability, `sm_120` in the compiled
  architectures, a matmul correctness check against the CPU, and a throughput benchmark.

## Consequences
- `uv sync` and CI stay fast and torch-free. GPU work needs an explicit `uv sync --group ml`.
- The cu128 index stops at torch 2.11. Newer releases publish wheels for newer CUDA versions.
  Upgrading past 2.11 means switching the index (the driver already supports CUDA 13.x).
- Production images (T13) will need their own torch decision (probably CPU wheels). This group
  does not cover them.
