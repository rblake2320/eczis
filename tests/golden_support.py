# UNFILED-IP
"""Shared inputs for golden fixtures (imported by tests and tools/regen_golden.py)."""

from __future__ import annotations

from pathlib import Path

from eczis.config import DemoConfig

GOLDEN_CFG = DemoConfig(sensor_height=24, sensor_width=32, frame_count=9, max_iterations=12)


def fixture_tree(root: Path) -> Path:
    (root / "nested").mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_bytes(b"golden evidence A\n")
    (root / "nested" / "b.bin").write_bytes(bytes(range(64)))
    return root
