"""Reliability diagrams and the coverage/accuracy curve, saved as PNG for MLflow."""

import importlib
from pathlib import Path
from typing import Any

from convintel.operators.intent.calibration import ReliabilityBin
from convintel.operators.intent.evaluation import CoveragePoint


def _pyplot() -> Any:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")  # headless: training runs in terminals and CI-like shells
    return importlib.import_module("matplotlib.pyplot")


def reliability_diagram(
    before: list[ReliabilityBin], after: list[ReliabilityBin], title: str, path: Path
) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, bins, name in ((axes[0], before, "before (T = 1)"), (axes[1], after, "after")):
        width = bins[0].upper - bins[0].lower
        filled = [b for b in bins if b.count]
        ax.bar(
            [b.lower for b in filled],
            [b.accuracy for b in filled],
            width=width,
            align="edge",
            edgecolor="black",
            label="accuracy",
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
        ax.plot([b.confidence for b in filled], [b.accuracy for b in filled], "o", color="C3")
        ax.set_title(name)
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def coverage_accuracy_plot(
    curves: dict[str, list[CoveragePoint]], threshold: float | None, target: float, path: Path
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, curve in curves.items():
        served = [p for p in curve if p.accuracy is not None]
        ax.plot([p.coverage for p in served], [p.accuracy for p in served], label=name)
        if threshold is not None:
            at = min(curve, key=lambda p: abs(p.threshold - threshold))
            if at.accuracy is not None:
                ax.plot(at.coverage, at.accuracy, "o", color="black")
    ax.axhline(target, linestyle="--", color="gray", label=f"target accuracy {target:.0%}")
    ax.set_xlabel("coverage (fraction served by the classifier)")
    ax.set_ylabel("accuracy on covered cases")
    label = f"τ = {threshold:.3f}" if threshold is not None else "no τ reaches the target"
    ax.set_title(f"Coverage vs accuracy ({label})")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
