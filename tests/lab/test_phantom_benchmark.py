# UNFILED-IP
"""Phantom benchmark: axis from the camera path, PCA refused when underdetermined, strict physical ordering."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "phantom_benchmark.py"
RGB = {"R": (37, 99, 235), "I": (245, 158, 11), "U": (120, 120, 120)}


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("phantom_benchmark", TOOL)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _cloud(length: float, theta_lo: float, theta_hi: float, r: float = 3.5, n: int = 3000, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    pts, cls = [], []
    for k, noise in (("R", 0.05), ("I", 0.3), ("U", 0.8)):
        t = rng.uniform(np.radians(theta_lo), np.radians(theta_hi), n)
        z = rng.uniform(0, length, n)
        rr = r + rng.normal(0, noise, n)
        pts.append(np.stack([rr * np.cos(t), rr * np.sin(t), z], axis=1))
        cls += [k] * n
    return np.vstack(pts), cls


def _write(tmp: Path, xyz: np.ndarray, cls: list[str], name: str = "graded.ply") -> Path:
    lines = [f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {RGB[c][0]} {RGB[c][1]} {RGB[c][2]}" for p, c in zip(xyz, cls, strict=True)]
    header = (f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    path = tmp / name
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def _traj(tmp: Path, length: float) -> Path:
    p = tmp / "traj.txt"
    p.write_text("\n".join(f"{i} 0.1 -0.2 {z:.3f} 0 0 0 1" for i, z in enumerate(np.linspace(0.5, length - 0.5, 8))) + "\n", encoding="utf-8")
    return p


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True, timeout=300)


def test_short_partial_patch_refuses_pca_axis(tmp_path: Path) -> None:
    xyz, cls = _cloud(length=3.0, theta_lo=-70, theta_hi=70)  # shorter than wide, 140 deg of wall: the ear-scan case
    r = _run(str(_write(tmp_path, xyz, cls)), "--diameter-mm", "7.0")
    assert r.returncode == 2 and "axis underdetermined" in r.stderr and "--trajectory" in r.stderr


def test_trajectory_axis_recovers_radius_and_orders_classes(tmp_path: Path) -> None:
    xyz, cls = _cloud(length=3.0, theta_lo=-70, theta_hi=70)
    r = _run(str(_write(tmp_path, xyz, cls)), "--diameter-mm", "7.0", "--trajectory", str(_traj(tmp_path, 3.0)), "--profile")
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "phantom_benchmark.json").read_text(encoding="utf-8"))
    assert out["axis_from"].startswith("camera trajectory")
    assert abs(out["fitted_R_radius_mm"] - 3.5) < 0.05
    c = out["classes"]
    assert c["R"]["median_abs_err_mm"] < 0.1 < c["I"]["median_abs_err_mm"] < 0.4 < c["U"]["median_abs_err_mm"]
    assert out["ordering_R<I<U"] is True
    assert "median_err_vs_R_profile_mm" in c["R"]  # the self-consistency column is extra, never the verdict


def test_long_wrapped_tube_allows_pca(tmp_path: Path) -> None:
    xyz, cls = _cloud(length=25.0, theta_lo=-180, theta_hi=180)
    r = _run(str(_write(tmp_path, xyz, cls)), "--diameter-mm", "7.0")
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "phantom_benchmark.json").read_text(encoding="utf-8"))
    assert out["axis_from"].startswith("PCA") and abs(out["fitted_R_radius_mm"] - 3.5) < 0.05


def test_metric_scale_changes_absolute_error(tmp_path: Path) -> None:
    xyz, cls = _cloud(length=3.0, theta_lo=-70, theta_hi=70)
    ply = _write(tmp_path, xyz * 0.5, cls)  # cloud in units where 1 unit = 2 mm
    traj = tmp_path / "traj.txt"
    traj.write_text("\n".join(f"{i} 0.05 -0.1 {z:.3f} 0 0 0 1" for i, z in enumerate(np.linspace(0.25, 1.25, 8))) + "\n", encoding="utf-8")
    r = _run(str(ply), "--diameter-mm", "7.0", "--trajectory", str(traj), "--scale-mm", "2.0", "--scale-units", "1.0")
    out = json.loads((tmp_path / "phantom_benchmark.json").read_text(encoding="utf-8"))
    assert r.returncode == 0 and out["mode"].startswith("absolute") and abs(out["fitted_R_radius_mm"] - 3.5) < 0.05


def test_ordering_requires_strict_improvement() -> None:
    mod = _load()
    assert mod.MIN_SINGULAR_RATIO >= 1.5
    # the verdict line in benchmark(): meds[0] < meds[1] < meds[2] -- ties must not pass
    src = TOOL.read_text(encoding="utf-8")
    assert "meds[0] < meds[1] < meds[2]" in src and "<=" not in src.split("ordering_R<I<U")[1].split("\n")[0]


@pytest.mark.parametrize("arc", [40.0])
def test_tiny_arc_is_refused(tmp_path: Path, arc: float) -> None:
    xyz, cls = _cloud(length=25.0, theta_lo=-arc / 2, theta_hi=arc / 2)
    r = _run(str(_write(tmp_path, xyz, cls)), "--diameter-mm", "7.0", "--trajectory", str(_traj(tmp_path, 25.0)))
    assert r.returncode == 2 and "circle fit is unstable" in r.stderr
