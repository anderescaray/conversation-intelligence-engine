import os
import time
from collections.abc import Callable
from pathlib import Path

from api_support import write_wav
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from convintel.api.audio import AudioSpool, sweep_spool

TTL = 3600.0
NOW = time.time()


def age(path: Path, seconds: float) -> Path:
    """Set the mtime of `path` itself (never a symlink's target) `seconds` in the past."""
    stamp = NOW - seconds
    os.utime(path, (stamp, stamp), follow_symlinks=False)
    return path


def test_discard_removes_only_regular_files_inside_the_spool(
    spool_dir: Path, tmp_path: Path
) -> None:
    spool = AudioSpool(spool_dir)
    inside = write_wav(spool_dir / "nested" / "a.wav")
    outside = write_wav(tmp_path / "outside.wav")
    link = spool_dir / "link.wav"
    link.symlink_to(outside)

    assert spool.discard(inside) is True
    assert not inside.exists()
    assert spool.discard(Path("nested/a.wav")) is False
    assert spool.discard(outside) is False
    assert spool.discard(spool_dir / ".." / "outside.wav") is False
    assert spool.discard(link) is False
    assert outside.exists()
    assert link.is_symlink()


def test_sweep_removes_expired_files_only(spool_dir: Path) -> None:
    old = age(write_wav(spool_dir / "old.wav"), TTL + 1)
    nested = age(write_wav(spool_dir / "calls" / "old.wav"), TTL + 1)
    fresh = age(write_wav(spool_dir / "fresh.wav"), TTL - 60)

    assert sweep_spool(AudioSpool(spool_dir), TTL, now=NOW) == 2
    assert not old.exists()
    assert not nested.exists()
    assert fresh.exists()
    assert (spool_dir / "calls").is_dir()


def test_sweep_never_follows_or_deletes_symlinks(spool_dir: Path, tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    target = age(write_wav(outside_dir / "victim.wav"), TTL * 10)
    file_link = spool_dir / "victim.wav"
    file_link.symlink_to(target)
    age(file_link, TTL * 10)
    dir_link = spool_dir / "linked-dir"
    dir_link.symlink_to(outside_dir, target_is_directory=True)
    age(dir_link, TTL * 10)

    assert sweep_spool(AudioSpool(spool_dir), TTL, now=NOW) == 0
    assert target.exists()
    assert file_link.is_symlink()
    assert dir_link.is_symlink()


def test_sweep_of_a_missing_spool_is_a_no_op(tmp_path: Path) -> None:
    assert sweep_spool(AudioSpool(tmp_path / "missing"), TTL) == 0


def test_startup_sweeps_the_spool(
    make_app: Callable[..., FastAPI], spool_dir: Path, spans: InMemorySpanExporter
) -> None:
    old = age(write_wav(spool_dir / "old.wav"), TTL + 1)
    fresh = write_wav(spool_dir / "fresh.wav")
    with TestClient(make_app(spool_ttl_s=TTL)):
        pass
    assert not old.exists()
    assert fresh.exists()
    (span,) = [s for s in spans.get_finished_spans() if s.name == "api.spool.sweep"]
    assert dict(span.attributes or {}) == {"files.removed": 1}
