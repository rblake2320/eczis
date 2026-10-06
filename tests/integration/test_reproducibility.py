# UNFILED-IP
"""Determinism, staged CLI, and bundle self-sufficiency.

Pre-registered expectations: two demo runs with the same seed produce identical
evidence; the staged CLI path runs end to end; the sealed bundle does NOT contain the
RAW observations, so provenance cannot be recomputed from the bundle alone (xfail).
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from eczis.bench.harness import run_demo
from eczis.config import DemoConfig


def test_same_seed_runs_are_bit_identical(demo_run: tuple[Path, dict[str, object]], tmp_path: Path) -> None:
    first, _ = demo_run
    second = tmp_path / "again"
    run_demo(second, DemoConfig(max_iterations=8))
    for name in ("provenance.npy", "confidence.npy", "map_reconstruction.npy", "challenge_provenance.npy"):
        a = np.load(first / "artifacts" / name)
        b = np.load(second / "artifacts" / name)
        assert np.array_equal(a, b), name
    m1 = json.loads((first / "artifacts" / "manifest.json").read_text())
    m2 = json.loads((second / "artifacts" / "manifest.json").read_text())
    assert m1["aggregate_sha256"] == m2["aggregate_sha256"]


def test_staged_cli_end_to_end(tmp_path: Path) -> None:
    def run(*args: str) -> None:
        proc = subprocess.run([sys.executable, "-m", "eczis.cli", *args], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-2000:]

    t = tmp_path / "target"
    cap = tmp_path / "capture.npz"
    rec = tmp_path / "recon"
    gate = tmp_path / "gate"
    prov = tmp_path / "prov"
    run("target", "create", "--output", str(t), "--seed", "20260803")
    run("capture", "simulate", "--target", str(t), "--output", str(cap), "--seed", "20260803", "--frames", "16")
    run("reconstruct", "--capture", str(cap), "--output", str(rec), "--iterations", "8")
    run("gate", "--capture", str(cap), "--reconstruction", str(rec / "map.npy"), "--output", str(gate), "--threshold", "8")
    run("provenance", "--capture", str(cap), "--gate", str(gate), "--output", str(prov))
    run("render", "--reconstruction", str(rec / "map.npy"), "--gate", str(gate), "--provenance", str(prov),
        "--output", str(tmp_path / "viewer.html"))
    assert (tmp_path / "viewer.html").stat().st_size > 1000


@pytest.mark.xfail(strict=True, reason="KNOWN GAP: bundle lacks RAW observations; classification not re-derivable")
def test_bundle_contains_observations_needed_to_recompute_provenance(
    demo_run: tuple[Path, dict[str, object]],
) -> None:
    output, _ = demo_run
    with zipfile.ZipFile(output / "eczis_demo.eczis") as archive:
        names = archive.namelist()
    assert any(n.endswith((".npz", "frames.npy", "raw.npy")) for n in names)
