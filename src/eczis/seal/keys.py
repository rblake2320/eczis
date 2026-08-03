# UNFILED-IP
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def create_key(path: Path) -> bytes:
    key = os.urandom(32)
    path.write_bytes(key)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key


def development_key(root_seed: int) -> bytes:
    """Deterministic demo-only key. Never use this for production evidence."""
    return hashlib.sha256(f"ECZIS-DEVELOPMENT-ONLY:{root_seed}".encode()).digest()
