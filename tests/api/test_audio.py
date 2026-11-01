from pathlib import Path

import pytest
from api_support import write_wav

from convintel.api.audio import AudioRejectedError, AudioSpool


def reason(spool: AudioSpool, uri: str) -> str:
    with pytest.raises(AudioRejectedError) as caught:
        spool.resolve(uri)
    return caught.value.reason


def test_accepts_relative_absolute_and_file_uri(spool_dir: Path) -> None:
    audio = write_wav(spool_dir / "calls" / "a.wav")
    spool = AudioSpool(spool_dir)
    assert spool.resolve("calls/a.wav") == audio.resolve()
    assert spool.resolve(str(audio)) == audio.resolve()
    assert spool.resolve(audio.resolve().as_uri()) == audio.resolve()


def test_rejects_paths_outside_the_spool(spool_dir: Path, tmp_path: Path) -> None:
    outside = write_wav(tmp_path / "outside.wav")
    (spool_dir / "link.wav").symlink_to(outside)
    spool = AudioSpool(spool_dir)
    assert reason(spool, "../outside.wav") == "outside_spool"
    assert reason(spool, str(outside)) == "outside_spool"
    assert reason(spool, "link.wav") == "outside_spool"
    assert reason(spool, outside.as_uri()) == "outside_spool"
    # Missing files outside the spool look the same as existing ones: no existence oracle.
    assert reason(spool, "../missing.wav") == "outside_spool"
    (spool_dir / "dangling.wav").symlink_to(tmp_path / "missing.wav")
    assert reason(spool, "dangling.wav") == "outside_spool"


@pytest.mark.parametrize(
    "uri",
    ["https://api.twilio.com/rec.wav", "file://remote-host/a.wav", "a.wav?x=1", "s3://b/a.wav"],
)
def test_rejects_non_local_uris(spool_dir: Path, uri: str) -> None:
    write_wav(spool_dir / "a.wav")
    assert reason(AudioSpool(spool_dir), uri) == "outside_spool"


def test_rejects_missing_files_and_directories(spool_dir: Path) -> None:
    (spool_dir / "dir.wav").mkdir()
    spool = AudioSpool(spool_dir)
    assert reason(spool, "missing.wav") == "not_found"
    assert reason(spool, "dir.wav") == "not_found"


def test_rejects_oversized_files_before_probing(spool_dir: Path) -> None:
    audio = write_wav(spool_dir / "a.wav")
    spool = AudioSpool(spool_dir, max_bytes=audio.stat().st_size - 1)
    assert reason(spool, "a.wav") == "too_large"


def test_rejects_audio_longer_than_the_limit(spool_dir: Path) -> None:
    write_wav(spool_dir / "a.wav", seconds=2.0)
    assert AudioSpool(spool_dir, max_duration_s=2.5).resolve("a.wav")
    assert reason(AudioSpool(spool_dir, max_duration_s=1.0), "a.wav") == "too_long"


def test_rejects_files_without_a_readable_duration(spool_dir: Path) -> None:
    (spool_dir / "noise.wav").write_bytes(b"not audio at all" * 64)
    assert reason(AudioSpool(spool_dir), "noise.wav") == "unreadable"


def test_check_duration_uses_the_same_limit(spool_dir: Path) -> None:
    spool = AudioSpool(spool_dir, max_duration_s=10)
    spool.check_duration(10)
    with pytest.raises(AudioRejectedError):
        spool.check_duration(10.5)
