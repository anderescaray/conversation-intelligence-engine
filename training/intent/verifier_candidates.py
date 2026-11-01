"""Candidate recall of the intent classifier below τ: the ceiling of the T10 LLM verifier.

    uv run --group ml python -m training.intent.verifier_candidates [--device cpu]

The verifier re-checks intents whose calibrated confidence is below τ and may only choose among
the classifier's top-k candidates, so it can never be right more often than the true label is
among them. This measures that ceiling (recall@k) on the below-τ cases, overall and restricted
to the cases the classifier gets wrong, plus a rough cost per k. It is a measurement: the model,
T and τ are the registered ones, loaded exactly as the operator loads them, and nothing here
feeds back into training or into the bundle.

Scored on the test split and on the validation split. Validation is where k should be chosen
(the README keeps test for final evaluation); test is the number the brief asks for.
"""

import argparse
import ast
import importlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from convintel.config import LLMProvider, get_settings
from convintel.llm.pricing import compute_cost
from convintel.operators.intent import DEFAULT_MODEL_URI, softmax
from convintel.operators.intent.model import DEFAULT_CACHE_DIR, load_bundle, resolve_model_uri
from convintel.operators.intent.operator import pick_intent

from .config import load_config
from .data import load_split, stratified_split

DEFAULT_STUDY_CONFIG = Path(__file__).resolve().parent / "verifier_candidates.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REPO_ROOT = Path(__file__).resolve().parents[2]


class TokenAssumptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fixed_prompt_tokens: int = Field(ge=0)
    chars_per_token: float = Field(gt=0)
    label_chars_per_token: float = Field(gt=0)
    per_candidate_overhead_tokens: int = Field(ge=0)
    conversation_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class VerifierStudyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ks: tuple[int, ...] = Field(min_length=1)
    taxonomy_commit: str
    taxonomy_path: str
    taxonomy_name: str
    llm_provider: LLMProvider
    llm_model: str
    tokens: TokenAssumptions


def load_study_config(path: Path = DEFAULT_STUDY_CONFIG) -> VerifierStudyConfig:
    return VerifierStudyConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def rank_candidates(
    probs: NDArray[np.floating], labels: Sequence[str], top_k: int
) -> tuple[NDArray[np.float64], list[tuple[str, ...]]]:
    """Confidence and top-k labels of each row, computed by the operator's own `pick_intent`.

    Each Banking77 example is one utterance, i.e. a conversation with one customer turn, so
    calling `pick_intent` row by row reproduces the served confidence and candidate order
    (including its tie-breaking) instead of re-implementing them.
    """
    outputs = [pick_intent(probs[i : i + 1], labels, top_k) for i in range(len(probs))]
    confidences = np.array([o.confidence for o in outputs], dtype=np.float64)
    return confidences, [tuple(label for label, _ in o.top_k) for o in outputs]


def below_threshold(confidences: NDArray[np.floating], tau: float) -> NDArray[np.bool_]:
    """Cases routed to the verifier: the classifier serves iff confidence ≥ τ (evaluation.py)."""
    return confidences < tau


def recall_at_k(candidates: Sequence[Sequence[str]], truth: Sequence[str], k: int) -> float | None:
    """Share of cases whose true label is among their first k candidates; None if no cases."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(candidates) != len(truth):
        raise ValueError("candidates and truth differ in length")
    if not truth:
        return None
    if any(len(c) < k for c in candidates):
        raise ValueError(f"fewer than {k} candidates for some case")
    return sum(t in c[:k] for c, t in zip(candidates, truth, strict=True)) / len(truth)


def accuracy_of(candidates: Sequence[Sequence[str]], truth: Sequence[str]) -> float | None:
    """Top-1 accuracy; None if no cases."""
    return recall_at_k(candidates, truth, 1)


def string_dicts(source: str) -> dict[str, dict[str, str]]:
    """Module-level dicts of string literals in `source`, with `**NAME` merges resolved.

    Reads the T06 taxonomy from another branch without importing or executing its code.
    Dicts holding anything else (non-string keys, calls, unknown names) are skipped.
    """
    resolved: dict[str, dict[str, str]] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, value = node.targets[0].id, node.value
        else:
            continue
        if not isinstance(value, ast.Dict):
            continue
        merged: dict[str, str] = {}
        for key, item in zip(value.keys, value.values, strict=True):
            if key is None and isinstance(item, ast.Name) and item.id in resolved:
                merged.update(resolved[item.id])
            elif (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(item, ast.Constant)
                and isinstance(item.value, str)
            ):
                merged[key.value] = item.value
            else:
                break
        else:
            resolved[name] = merged
    return resolved


def candidate_tokens(
    names: Sequence[str], descriptions: dict[str, str], tokens: TokenAssumptions
) -> float:
    """Estimated prompt tokens for listing `names` with their descriptions."""
    return sum(
        len(name) / tokens.label_chars_per_token
        + len(descriptions[name]) / tokens.chars_per_token
        + tokens.per_candidate_overhead_tokens
        for name in names
    )


class CostRow(BaseModel):
    k: int
    calls: int
    mean_input_tokens_utterance: float
    usd_utterance: float
    """One verifier pass over the below-τ cases, prompt carrying the Banking77 utterance."""
    mean_input_tokens_conversation: float
    usd_conversation: float
    """Same calls, prompt carrying a whole conversation of `conversation_tokens`."""
    usd_per_1000_conversations: float
    """Conversation scenario times the below-τ share: verifier spend per 1 000 conversations."""


def estimate_cost(
    candidates: Sequence[Sequence[str]],
    texts: Sequence[str],
    k: int,
    share_below: float,
    descriptions: dict[str, str],
    tokens: TokenAssumptions,
    provider: LLMProvider,
    model: str,
) -> CostRow:
    """Cost of one verifier call per below-τ case, priced by the LLM client's own table."""
    calls = len(texts)
    shared = [
        tokens.fixed_prompt_tokens + candidate_tokens(c[:k], descriptions, tokens)
        for c in candidates
    ]
    utterance = [s + len(t) / tokens.chars_per_token for s, t in zip(shared, texts, strict=True)]
    conversation = [s + tokens.conversation_tokens for s in shared]
    output = tokens.output_tokens * calls

    def usd(input_tokens: list[float]) -> float:
        cost = compute_cost(provider, model, round(sum(input_tokens)), output)
        if cost is None:
            raise ValueError(f"no price for {provider}:{model}")
        return cost

    usd_conversation = usd(conversation)
    return CostRow(
        k=k,
        calls=calls,
        mean_input_tokens_utterance=float(np.mean(utterance)) if calls else 0.0,
        usd_utterance=usd(utterance),
        mean_input_tokens_conversation=float(np.mean(conversation)) if calls else 0.0,
        usd_conversation=usd_conversation,
        usd_per_1000_conversations=usd_conversation / calls * share_below * 1000 if calls else 0.0,
    )


class RecallRow(BaseModel):
    k: int
    below_tau: float | None
    below_tau_wrong: float | None


class SplitReport(BaseModel):
    split: str
    n: int
    tau: float
    accuracy: float
    coverage: float
    n_below: int
    share_below: float
    accuracy_below: float | None
    """The baseline the verifier must beat on the cases it sees."""
    accuracy_above: float | None
    n_below_wrong: int
    recall: list[RecallRow]
    cost: list[CostRow]


def study_split(
    split: str,
    confidences: NDArray[np.floating],
    candidates: Sequence[Sequence[str]],
    truth: Sequence[str],
    texts: Sequence[str],
    tau: float,
    config: VerifierStudyConfig,
    descriptions: dict[str, str],
) -> SplitReport:
    below = below_threshold(confidences, tau)
    cand_below = [c for c, b in zip(candidates, below, strict=True) if b]
    truth_below = [t for t, b in zip(truth, below, strict=True) if b]
    texts_below = [x for x, b in zip(texts, below, strict=True) if b]
    cand_above = [c for c, b in zip(candidates, below, strict=True) if not b]
    truth_above = [t for t, b in zip(truth, below, strict=True) if not b]
    wrong = [c[0] != t for c, t in zip(cand_below, truth_below, strict=True)]
    cand_wrong = [c for c, w in zip(cand_below, wrong, strict=True) if w]
    truth_wrong = [t for t, w in zip(truth_below, wrong, strict=True) if w]
    n, n_below = len(truth), int(below.sum())
    overall = accuracy_of(candidates, truth)
    if overall is None:
        raise ValueError(f"empty {split} split")
    return SplitReport(
        split=split,
        n=n,
        tau=tau,
        accuracy=overall,
        coverage=1 - n_below / n,
        n_below=n_below,
        share_below=n_below / n,
        accuracy_below=accuracy_of(cand_below, truth_below),
        accuracy_above=accuracy_of(cand_above, truth_above),
        n_below_wrong=len(truth_wrong),
        recall=[
            RecallRow(
                k=k,
                below_tau=recall_at_k(cand_below, truth_below, k),
                below_tau_wrong=recall_at_k(cand_wrong, truth_wrong, k),
            )
            for k in config.ks
        ],
        cost=[
            estimate_cost(
                cand_below,
                texts_below,
                k,
                n_below / n,
                descriptions,
                config.tokens,
                config.llm_provider,
                config.llm_model,
            )
            for k in config.ks
        ],
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f} %"


def render_markdown(report: SplitReport) -> str:
    """Results tables of one split, as pasted into reports/verifier-candidates.md."""
    lines = [
        f"### {report.split} (n = {report.n}, τ = {report.tau})",
        "",
        "| | Value |",
        "|---|---|",
        f"| Below τ | {report.n_below} ({_pct(report.share_below)}) |",
        f"| Classifier accuracy below τ (baseline) | {_pct(report.accuracy_below)} |",
        f"| Classifier accuracy at or above τ | {_pct(report.accuracy_above)} |",
        f"| Below τ and wrong | {report.n_below_wrong} |",
        "",
        "| k | recall@k below τ | recall@k below τ, classifier wrong "
        "| mean input tokens (utterance / conversation) | USD one pass (utterance / conversation) "
        "| USD per 1 000 conversations |",
        "|---|---|---|---|---|---|",
    ]
    for r, c in zip(report.recall, report.cost, strict=True):
        lines.append(
            f"| {r.k} | {_pct(r.below_tau)} | {_pct(r.below_tau_wrong)} "
            f"| {c.mean_input_tokens_utterance:.0f} / {c.mean_input_tokens_conversation:.0f} "
            f"| {c.usd_utterance:.3f} / {c.usd_conversation:.3f} "
            f"| {c.usd_per_1000_conversations:.3f} |"
        )
    return "\n".join(lines)


def read_descriptions(config: VerifierStudyConfig) -> dict[str, str]:
    """Taxonomy descriptions from the pinned commit (local git objects only, no fetch)."""
    source = subprocess.run(
        ["git", "show", f"{config.taxonomy_commit}:{config.taxonomy_path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dicts = string_dicts(source)
    if config.taxonomy_name not in dicts:
        raise ValueError(f"{config.taxonomy_name} not found in {config.taxonomy_path}")
    return dicts[config.taxonomy_name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Candidate recall@k below τ for the verifier.")
    parser.add_argument("--config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    config = load_study_config(args.config)
    train_cfg = load_config()
    descriptions = read_descriptions(config)
    directory, model_id = resolve_model_uri(
        args.model_uri, args.cache_dir, get_settings().mlflow_tracking_uri
    )
    model = load_bundle(directory, model_id, device=args.device)
    if model.threshold is None:
        raise ValueError(f"{model_id} ships no τ")
    missing = set(model.labels) - set(descriptions)
    if missing:
        raise ValueError(f"no description for {sorted(missing)}")

    train = load_split("train")
    _, val_idx = stratified_split(train.labels, train_cfg.val_fraction, train_cfg.seed)
    splits = {"validation": train.subset(val_idx), "test": load_split("test")}

    reports: list[SplitReport] = []
    for name, data in splits.items():
        if tuple(data.label_names) != tuple(model.labels):
            raise ValueError(f"{name} label ids do not match the model's label order")
        probs = softmax(model.predict_logits(data.texts), model.temperature)
        confidences, candidates = rank_candidates(probs, model.labels, max(config.ks))
        truth = [data.label_names[i] for i in data.labels.tolist()]
        reports.append(
            study_split(
                name,
                confidences,
                candidates,
                truth,
                data.texts,
                model.threshold,
                config,
                descriptions,
            )
        )

    torch = importlib.import_module("torch")
    payload = {
        "model_id": model_id,
        "temperature": model.temperature,
        "tau": model.threshold,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "config": config.model_dump(),
        "splits": [r.model_dump() for r in reports],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_dir / f"verifier-candidates-{stamp}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"model {model_id}, T = {model.temperature:.4f}, τ = {model.threshold}; wrote {out}\n")
    for r in reports:
        print(render_markdown(r))
        print(f"\n(accuracy {r.accuracy:.6f}, coverage at τ {r.coverage:.6f})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
