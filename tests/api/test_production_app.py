"""The production composition root: what uvicorn serves with `--factory`."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from api_support import FakePipeline, isolate_spool
from fastapi.testclient import TestClient

from convintel.api import main
from convintel.config import Settings


def test_production_app_serves_the_pipeline_built_from_settings(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, pipeline: FakePipeline
) -> None:
    built_with: list[Settings] = []

    def fake_build_pipeline(given: Settings) -> FakePipeline:
        built_with.append(given)
        return pipeline

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "build_pipeline", fake_build_pipeline)

    app = main.create_production_app()

    # One Settings object for the pipeline and the API (Twilio auth reads it from the app).
    assert built_with == [settings]
    assert app.state.convintel.pipeline is pipeline
    assert app.state.convintel.settings is settings


def test_isolate_spool_moves_the_startup_sweep_away_from_the_default_spool(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, pipeline: FakePipeline, spool_dir: Path
) -> None:
    # The helper the end-to-end test relies on, checked here for free (no models, no LLM).
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "build_pipeline", lambda _: pipeline)
    app = main.create_production_app()
    app.state.convintel.telemetry_setup = lambda: None
    stale = spool_dir / "stale.wav"
    stale.write_bytes(b"synthetic")
    os.utime(stale, (0, 0))

    isolate_spool(app, spool_dir)
    with TestClient(app):
        pass

    # Only a sweep of the test spool deletes it: the default spool is somewhere else.
    assert app.state.convintel.spool.root == spool_dir.resolve()
    assert not stale.exists()


def _run_python(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    # A fresh interpreter: this process has already imported the module. Run from an empty
    # directory so no local .env is read, with a setting that cannot validate.
    env = {**os.environ, "INTENT_VERIFIER_THRESHOLD": "86"}
    return subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


def test_importing_the_api_reads_no_settings(tmp_path: Path) -> None:
    imported = _run_python("import convintel.api.main", tmp_path)
    assert imported.returncode == 0, imported.stderr

    # Not vacuous: the same environment does fail as soon as the app is built.
    started = _run_python(
        "from convintel.api.main import create_production_app; create_production_app()", tmp_path
    )
    assert started.returncode != 0
    assert "intent_verifier_threshold" in started.stderr
