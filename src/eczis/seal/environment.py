# UNFILED-IP
"""Replay metadata recorded in every manifest (Phase 8)."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any


def _lock_digest() -> str | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "uv.lock"
        if candidate.is_file():
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
    return None


def replay_environment() -> dict[str, Any]:
    import numpy as np
    import scipy

    from eczis import __version__

    return {
        "eczis_version": __version__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "byteorder": sys.byteorder,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "uv_lock_sha256": _lock_digest(),
        "replay_policy": "same-platform bit-identical where feasible; cross-platform by declared tolerance",
    }
