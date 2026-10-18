"""Gate a judge on hand-written controls before it may filter any pilot.

Each control states what its text truly shows (`truth`) and the labels it is checked against
(`targets`), which differ in at most the one `broken` field. A judge passes only if it flags
nothing on valid controls and exactly the broken field on the others. With ten controls this is
a gate, not an accuracy estimate, so results are reported as counts.
"""

import argparse
import asyncio
import os
import time
from pathlib import Path

from pydantic import Field, model_validator

from convintel.config import Settings
from convintel.errors import ConvIntelError, LLMError
from convintel.llm import LLMClient, load_prompt
from convintel.telemetry import traced

from .judge import JUDGE_PROMPT_VERSION, configured_client, evidence_errors, observe
from .scenarios import INTENTS
from .schema import (
    LABELS,
    VERSION,
    DraftTurn,
    JudgeObservation,
    Label,
    LabelTargets,
    Record,
    compare,
)
from .storage import digest, file_hash, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTROLS = ROOT / "tests/synth/judge_controls.jsonl"


class Control(Record):
    id: str
    note: str
    broken: Label | None
    truth: LabelTargets
    targets: LabelTargets
    turns: tuple[DraftTurn, ...] = Field(min_length=6, max_length=10)

    @model_validator(mode="after")
    def validate_control(self) -> "Control":
        if self.truth.intent not in INTENTS or self.targets.intent not in INTENTS:
            raise ValueError("unknown intent")
        differing = {
            label for label in LABELS if getattr(self.truth, label) != getattr(self.targets, label)
        }
        if differing != ({self.broken} if self.broken else set()):
            raise ValueError("targets must differ from truth in exactly the broken field")
        for index, turn in enumerate(self.turns):
            if turn.speaker != ("customer" if index % 2 == 0 else "agent"):
                raise ValueError("controls alternate customer and agent turns")
        return self


class ControlResult(Record):
    id: str
    broken: Label | None
    flagged: tuple[Label, ...]
    wrong_fields: tuple[Label, ...]
    evidence_errors: tuple[str, ...]
    observation: JudgeObservation | None
    error: str | None = None
    ok: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


class GateReport(Record):
    generator_version: str = VERSION
    judge_model_id: str
    judge_model_revision: str
    prompt_version: str
    prompt_sha256: str
    controls_sha256: str
    controls: int
    passed_controls: str
    field_correct: dict[str, str]
    gate_passed: bool
    elapsed_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    results: list[ControlResult]


def load_controls(path: Path) -> list[Control]:
    controls = read_jsonl(path, Control)
    if not controls or len({control.id for control in controls}) != len(controls):
        raise ConvIntelError("Controls must be a nonempty set with unique IDs")
    return controls


def check_control(control: Control, observation: JudgeObservation) -> ControlResult:
    """Score one observation; usage fields are filled in by the caller."""
    flagged = compare(observation, control.targets)
    errors = evidence_errors(observation, [turn.speaker for turn in control.turns])
    expected = (control.broken,) if control.broken else ()
    return ControlResult(
        id=control.id,
        broken=control.broken,
        flagged=flagged,
        wrong_fields=compare(observation, control.truth),
        evidence_errors=errors,
        observation=observation,
        ok=flagged == expected and not errors,
        latency_ms=0,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0,
    )


def summarise(
    results: list[ControlResult],
    *,
    judge_model_id: str,
    judge_model_revision: str,
    prompt_version: str,
    prompt_sha256: str,
    controls_sha256: str,
    elapsed_s: float,
) -> GateReport:
    total = len(results)
    correct = {
        label: sum(r.observation is not None and label not in r.wrong_fields for r in results)
        for label in LABELS
    }
    passed = sum(result.ok for result in results)
    return GateReport(
        judge_model_id=judge_model_id,
        judge_model_revision=judge_model_revision,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
        controls_sha256=controls_sha256,
        controls=total,
        passed_controls=f"{passed}/{total}",
        field_correct={label: f"{count}/{total}" for label, count in correct.items()},
        gate_passed=passed == total,
        elapsed_s=elapsed_s,
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens for r in results),
        cost_usd=None
        if any(r.cost_usd is None for r in results)
        else sum(r.cost_usd or 0 for r in results),
        results=results,
    )


async def run_gate(
    client: LLMClient, controls_path: Path, *, model_revision: str = "unresolved"
) -> GateReport:
    prompt = await asyncio.to_thread(load_prompt, "synthetic_judge", JUDGE_PROMPT_VERSION)
    controls = await asyncio.to_thread(load_controls, controls_path)
    results: list[ControlResult] = []
    start = time.perf_counter()
    with traced("synthetic.judge_check", version=VERSION) as span:
        span.set_attribute("model.id", client.model_id)
        for control in controls:
            try:
                response = await observe(client, prompt, control.turns, conversation_id=control.id)
            except LLMError as exc:
                # A judge that cannot answer a control fails it; the gate never retries silently.
                results.append(
                    ControlResult(
                        id=control.id,
                        broken=control.broken,
                        flagged=(),
                        wrong_fields=LABELS,
                        evidence_errors=(),
                        observation=None,
                        error=type(exc).__name__,
                        ok=False,
                        latency_ms=0,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=None if not client.model_id.startswith("ollama:") else 0,
                    )
                )
                continue
            results.append(
                check_control(control, response.output).model_copy(
                    update={
                        "latency_ms": response.latency_ms,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    }
                )
            )
        report = summarise(
            results,
            judge_model_id=client.model_id,
            judge_model_revision=model_revision,
            prompt_version=prompt.id,
            prompt_sha256=digest(prompt),
            controls_sha256=await asyncio.to_thread(file_hash, controls_path),
            elapsed_s=time.perf_counter() - start,
        )
        span.set_attribute("judge.gate_passed", report.gate_passed)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the judge control gate")
    parser.add_argument("--provider", choices=("ollama", "anthropic"), default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--model-revision", default="unresolved")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["OPENINFERENCE_HIDE_INPUTS"] = "true"
    os.environ["OPENINFERENCE_HIDE_OUTPUTS"] = "true"
    if args.output.exists():
        parser.exit(1, "Output exists; gate reports are immutable\n")
    settings = Settings(
        llm_provider=args.provider,
        llm_model=args.model or ("qwen3:8b" if args.provider == "ollama" else "claude-haiku-4-5"),
    )
    try:
        client = configured_client(settings, "synthetic_judge", JUDGE_PROMPT_VERSION)
        report = asyncio.run(run_gate(client, args.fixtures, model_revision=args.model_revision))
    except ConvIntelError as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(report.model_dump_json(indent=2, exclude={"results"}))
    for result in report.results:
        print(
            f"{result.id}: ok={result.ok} broken={result.broken} flagged={list(result.flagged)} "
            f"wrong={list(result.wrong_fields)} evidence_errors={list(result.evidence_errors)}"
        )
    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
