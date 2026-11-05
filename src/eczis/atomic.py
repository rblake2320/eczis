# UNFILED-IP
"""Crash-safe file primitives.

Every evidence-bearing write goes through ``atomic_write_bytes``: data is
written to a sibling temporary file, flushed, fsynced, and moved into place
with ``os.replace``. A crash leaves either the previous file or the complete
new file -- never a torn file under the final name.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":  # Windows cannot open a directory handle for fsync.
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes, *, mode: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            with contextlib.suppress(OSError):
                temporary.chmod(mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def exclusive_create_bytes(path: Path, data: bytes, *, mode: int = 0o600) -> Path:
    """Create ``path`` only if it does not exist; never overwrite (key material)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)
    return path


@contextlib.contextmanager
def file_lock(lock_path: Path, *, timeout_s: float = 30.0) -> Iterator[IO[bytes]]:
    """Cross-process exclusive lock (msvcrt on Windows, fcntl on POSIX)."""
    import time

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")  # noqa: SIM115 - lifetime managed by the context manager
    deadline = time.monotonic() + timeout_s
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"could not lock {lock_path}") from None
                    time.sleep(0.01)
            try:
                yield handle
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"could not lock {lock_path}") from None
                    time.sleep(0.01)
            try:
                yield handle
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
