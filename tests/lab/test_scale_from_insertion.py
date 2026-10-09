# UNFILED-IP
"""Scale-from-insertion: a known probe movement turns reconstruction units into millimetres."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "scale_from_insertion.py"


def _write_traj(folder: Path, positions: np.ndarray) -> None:
    (folder / "mast3r").mkdir(parents=True)
    lines = [f"{i:.3f} {p[0]} {p[1]} {p[2]} 0 0 0 1" for i, p in enumerate(positions)]
    (folder / "mast3r" / "sweep.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_straight_insertion_recovers_scale(tmp_path: Path) -> None:
    # camera advanced 0.4 units straight along z while the probe was pushed 5 mm
    _write_traj(tmp_path, np.array([[0, 0, 0.1 * i] for i in range(5)], dtype=float))
    r = subprocess.run([sys.executable, str(TOOL), str(tmp_path), "--moved-mm", "5.0", "--single"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads((tmp_path / "scale.json").read_text(encoding="utf-8"))
    assert abs(out["moved_units"] - 0.4) < 1e-9
    assert abs(out["scale_units_to_mm"] - 12.5) < 1e-6
    assert out["path_over_straight"] == 1.0
    assert "--scale-units 0.4" in r.stdout


def test_wobbly_path_is_flagged(tmp_path: Path) -> None:
    z = np.linspace(0, 0.4, 9)
    _write_traj(tmp_path, np.stack([0.3 * np.sin(np.arange(9) * 2.0), np.zeros(9), z], axis=1))
    r = subprocess.run([sys.executable, str(TOOL), str(tmp_path), "--moved-mm", "5.0"],
                       capture_output=True, text=True, timeout=120)
    out = json.loads((tmp_path / "scale.json").read_text(encoding="utf-8"))
    assert out["path_over_straight"] > 1.3  # the caveat threshold the output names
    assert r.returncode == 0  # still a number, but the record carries the wobble ratio


def test_two_sweeps_refuse_a_scale(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _write_traj(a, np.array([[0, 0, 0.1 * i] for i in range(4)], dtype=float))
    _write_traj(b, np.array([[0, 0, 0.2 * i] for i in range(4)], dtype=float))
    r = subprocess.run([sys.executable, str(TOOL), str(a), str(b), "--moved-mm", "5.0"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert json.loads((a / "scale.json").read_text(encoding="utf-8"))["scale_units_to_mm"] is None


def test_missing_trajectory_is_explicit(tmp_path: Path) -> None:
    r = subprocess.run([sys.executable, str(TOOL), str(tmp_path), "--moved-mm", "5.0"], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "no mast3r/*.txt" in r.stderr


def test_tool_is_importable_module() -> None:
    spec = importlib.util.spec_from_file_location("scale_from_insertion", TOOL)
    assert spec is not None
