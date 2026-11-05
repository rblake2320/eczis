# UNFILED-IP
"""Seal-key handling.

- ``create_key`` writes 32 random bytes with exclusive-create (never
  overwrites an existing key) and owner-only permissions where the OS
  honours POSIX modes.
- On Windows, ``protect=True`` wraps the key with DPAPI (CurrentUser scope)
  so the file is useless when copied to another account or machine.
- ``development_key`` is deterministic and for synthetic demos only.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from eczis.atomic import exclusive_create_bytes

KEY_BYTES = 32
MIN_KEY_BYTES = 32
DPAPI_MAGIC = b"ECZIS-DPAPI-v1\n"
_ENTROPY = b"ECZIS seal key v1"


class KeyMaterialError(ValueError):
    """Raised for unusable key material."""


def dpapi_available() -> bool:
    return sys.platform == "win32"


def _dpapi(data: bytes, *, protect: bool) -> bytes:  # pragma: no cover - exercised on Windows CI
    if sys.platform != "win32":  # narrows for mypy: the body below is type-checked on Windows only
        raise OSError("DPAPI is only available on Windows")
    import ctypes
    from ctypes import wintypes

    class DataBlob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def _blob(payload: bytes) -> tuple[DataBlob, ctypes.Array[ctypes.c_char]]:
        buffer = ctypes.create_string_buffer(payload, len(payload))
        return DataBlob(len(payload), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    data_in, _keep_in = _blob(data)
    entropy, _keep_entropy = _blob(_ENTROPY)
    data_out = DataBlob()
    flags = 0x1  # CRYPTPROTECT_UI_FORBIDDEN
    if protect:
        ok = crypt32.CryptProtectData(
            ctypes.byref(data_in), "ECZIS seal key", ctypes.byref(entropy), None, None, flags,
            ctypes.byref(data_out),
        )
    else:
        ok = crypt32.CryptUnprotectData(
            ctypes.byref(data_in), None, ctypes.byref(entropy), None, None, flags, ctypes.byref(data_out)
        )
    if not ok:
        raise OSError(ctypes.get_last_error(), "DPAPI call failed")
    try:
        return ctypes.string_at(data_out.pbData, data_out.cbData)
    finally:
        kernel32.LocalFree(data_out.pbData)


def create_key(path: Path, *, protect: bool = False) -> bytes:
    """Generate a new key file. Refuses to overwrite an existing file."""
    key = os.urandom(KEY_BYTES)
    payload = DPAPI_MAGIC + _dpapi(key, protect=True) if protect else key
    exclusive_create_bytes(path, payload, mode=0o600)
    return key


def load_key(path: Path) -> bytes:
    """Load a raw or DPAPI-wrapped key; fail closed on short or unreadable material."""
    payload = path.read_bytes()
    if payload.startswith(DPAPI_MAGIC):
        key = _dpapi(payload[len(DPAPI_MAGIC) :], protect=False)
    else:
        key = payload
    if len(key) < MIN_KEY_BYTES:
        raise KeyMaterialError(f"key material in {path.name} is {len(key)} bytes; need >= {MIN_KEY_BYTES}")
    return key


def development_key(root_seed: int) -> bytes:
    """Deterministic demo-only key. Never use this for production evidence."""
    return hashlib.sha256(f"ECZIS-DEVELOPMENT-ONLY:{root_seed}".encode()).digest()
