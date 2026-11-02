"""The production composition root: what uvicorn serves with `--factory`."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from api_support import FakePipeline

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
