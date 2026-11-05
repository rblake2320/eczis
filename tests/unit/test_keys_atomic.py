# UNFILED-IP
from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from eczis import atomic
from eczis.atomic import atomic_write_bytes, exclusive_create_bytes
from eczis.seal.keys import DPAPI_MAGIC, KeyMaterialError, create_key, dpapi_available, load_key

PLATFORM_NA = "platform-not-applicable"


def test_create_key_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "seal.key"
    key = create_key(path, protect=False)
    assert len(key) == 32
    assert load_key(path) == key
    with pytest.raises(FileExistsError):
        create_key(path, protect=False)
    assert load_key(path) == key  # original key untouched


@pytest.mark.skipif(sys.platform == "win32", reason=f"{PLATFORM_NA}: POSIX permission bits")
def test_key_file_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "seal.key"
    create_key(path, protect=False)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_short_key_material_rejected(tmp_path: Path) -> None:
    path = tmp_path / "short.key"
    path.write_bytes(b"0123456789")
    with pytest.raises(KeyMaterialError, match="need >= 32"):
        load_key(path)


@pytest.mark.skipif(not dpapi_available(), reason=f"{PLATFORM_NA}: DPAPI is Windows-only")
def test_dpapi_round_trip_and_blob_is_not_raw_key(tmp_path: Path) -> None:
    path = tmp_path / "dpapi.key"
    key = create_key(path, protect=True)
    blob = path.read_bytes()
    assert blob.startswith(DPAPI_MAGIC)
    assert key not in blob
    assert load_key(path) == key


@pytest.mark.skipif(dpapi_available(), reason=f"{PLATFORM_NA}: checks non-Windows refusal")
def test_dpapi_refused_off_windows(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Windows"):
        create_key(tmp_path / "k.key", protect=True)
    assert not (tmp_path / "k.key").exists()


def test_atomic_write_preserves_old_file_on_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "state.json"
    atomic_write_bytes(target, b"old")

    def boom(*_: object) -> None:
        raise OSError("crash")

    monkeypatch.setattr("eczis.atomic.os.replace", boom)
    with pytest.raises(OSError):
        atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_exclusive_create_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "once.bin"
    exclusive_create_bytes(target, b"first")
    with pytest.raises(FileExistsError):
        exclusive_create_bytes(target, b"second")
    assert target.read_bytes() == b"first"


def test_file_lock_times_out_when_held(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_hold_lock, args=(str(lock), ready, release))
    proc.start()
    try:
        assert ready.wait(30)
        with pytest.raises(TimeoutError), atomic.file_lock(lock, timeout_s=0.3):
            pass
    finally:
        release.set()
        proc.join(30)
    with atomic.file_lock(lock, timeout_s=5):
        pass


def _hold_lock(path: str, ready: object, release: object) -> None:
    with atomic.file_lock(Path(path)):
        ready.set()  # type: ignore[attr-defined]
        release.wait(30)  # type: ignore[attr-defined]


