# UNFILED-IP
"""The physical-benchmark protocol's records and controls: glare refusal, print record, phantom record,
policy freeze (blind analysis), calibration held-out/diversity/perturbation, and the report assembly."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def _mod(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------- glare / dark refusal
def test_glare_sweep_is_refused_before_any_geometry() -> None:
    from eczis.lab.scan3d import SweepRefused, require_usable, sweep_quality

    good = [np.full((24, 32), 0.5) for _ in range(10)]
    glare = [np.where(np.arange(32)[None, :] < 4, 1.0, 0.5) for _ in range(10)]  # 12.5 % clipped-bright
    dark = [np.full((24, 32), 0.01) for _ in range(10)]
    assert require_usable(good)["refused"] is False
    assert sweep_quality(glare)["reason"] == "GLARE"
    with pytest.raises(SweepRefused, match="GLARE"):
        require_usable(glare)
    with pytest.raises(SweepRefused, match="TOO DARK"):
        require_usable(dark)


def test_ear_scan_analyse_writes_refusal_record(tmp_path: Path) -> None:
    sys.path.insert(0, str(TOOLS))
    ear = _mod("ear_scan")
    from eczis.lab.scan3d import SweepRefused

    frames = [np.where(np.arange(64)[None, :] < 8, 1.0, 0.5) for _ in range(20)]
    with pytest.raises(SweepRefused):
        ear.analyse(frames, tmp_path)
    rec = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert rec["status"].startswith("REFUSED: GLARE") and rec["statuses"]["geometry_support"].startswith("NONE")
    assert not (tmp_path / "scan.ply").exists()


# ---------------------------------------------------------------- print record
def test_print_record_axis_mismatch_is_refused(tmp_path: Path) -> None:
    pr = _mod("print_record")
    bad = pr.record("p", "paper", "v", "100%", 20.0, 20.5, path=tmp_path / "pr.json")
    assert bad["status"].startswith("MISMATCH") and bad["axis_mismatch_frac"] == 0.025
    good = pr.record("p", "paper", "v", "100%", 20.1, 20.1, path=tmp_path / "pr2.json")
    assert good["status"] == "OK" and abs(good["effective_square_mm"] - 2.01) < 1e-6
    r = subprocess.run([sys.executable, str(TOOLS / "print_record.py"), "--printer", "p", "--paper", "a", "--viewer", "v", "--scaling", "s",
                        "--h-mm", "20", "--v-mm", "21", "--out", str(tmp_path / "x.json")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2


# ---------------------------------------------------------------- phantom record
def test_phantom_record_binds_frames_hash(tmp_path: Path) -> None:
    pr = _mod("phantom_record")
    np.savez_compressed(tmp_path / "frames.npz", frames=np.zeros((3, 4, 4), np.float32))
    rec = pr.record(tmp_path, role="straight", diameter_mm=8.0)
    saved = json.loads((tmp_path / "phantom_record.json").read_text(encoding="utf-8"))
    assert saved["role"] == "straight" and len(saved["frames_sha256"]) == 64 and saved == rec


# ---------------------------------------------------------------- policy freeze
def test_policy_freeze_detects_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from eczis.lab import policy_freeze as pf

    p = tmp_path / "freeze.json"
    assert pf.check(p)["status"] == "NOT_FROZEN"
    pf.freeze(p, note="test")
    assert pf.check(p)["status"] == "BLIND"
    from eczis.lab import scan3d

    monkeypatch.setattr(scan3d, "MAX_GLARE_FRAC", 0.5)  # someone loosens a threshold after seeing results
    st = pf.check(p)
    assert st["status"] == "CHANGED" and "scan3d.MAX_GLARE_FRAC" in st["changed"]


# ---------------------------------------------------------------- calibration held-out / diversity / perturbation
def test_calibration_held_out_and_perturbation() -> None:
    from eczis.lab.calibrate import Calibrator, synthetic_views

    k = np.array([[520.0, 0, 322.0], [0, 518.0, 238.0], [0, 0, 1.0]])
    d = np.array([-0.28, 0.09, 0.0005, -0.0004, 0.0])
    cal = Calibrator()
    for v in synthetic_views(k, d, (640, 480), count=30, seed=1):
        cal.add(v)
    prof = cal.solve(camera="syn", capture_mode="t")
    assert len(prof.held_out_rms_px) == prof.views >= 8
    assert prof.held_out_stable() is True and max(prof.held_out_rms_px) < 1.5
    assert {"centre_spread_norm", "tilt_deg_max", "size_ratio_max_over_min"} <= set(prof.diversity)
    assert isinstance(prof.diverse(), bool)
    bad = prof.perturbed(0.10)
    assert abs(bad.fx / prof.fx - 1.10) < 1e-9 and bad.cx == prof.cx + 32.0 and "PERTURBED" in bad.source
    assert prof.perturbed(0.0).fx == prof.fx


# ---------------------------------------------------------------- report assembly
def _bench(r: float, i: float, u: float, fitted: float = 4.0, arc: float = 200.0, blind: str = "BLIND", absolute: bool = True) -> dict[str, Any]:
    cls = {k: {"n": n, "median_abs_err_mm": m, "p90_abs_err_mm": m * 2, "p95_abs_err_mm": m * 3, "frac_within_0.5mm": 0.9}
           for k, m, n in (("R", r, 1000), ("I", i, 500), ("U", u, 700))}
    return {"mode": "absolute (metric scale supplied)" if absolute else "RELATIVE", "axis_from": "camera trajectory", "true_radius_mm": 4.0,
            "fitted_R_radius_mm": fitted, "R_arc_coverage_deg": arc, "classes": cls, "ordering_R<I<U": r < i < u,
            "ordering_R<I<U_p90": r < i < u, "ordering_R<I<U_p95": r < i < u, "blind_analysis": {"status": blind},
            "profile_vs_global_disagreement_mm": 0.1, "diameter_error_by_slice": [{"z_mm": 1.0, "R_diameter_mm": 8.0, "error_mm": 0.0}]}


def _sweep(tmp: Path, name: str, role: str, bench: dict[str, Any] | None, result: dict[str, Any] | None,
           perturbed: dict[str, Any] | None = None, summary: dict[str, Any] | None = None) -> Path:
    f = tmp / name
    (f / "mast3r").mkdir(parents=True)
    (f / "phantom_record.json").write_text(json.dumps({"role": role, "diameter_mm": 8.0}), encoding="utf-8")
    if bench:
        (f / "mast3r" / "phantom_benchmark.json").write_text(json.dumps(bench), encoding="utf-8")
    if result:
        (f / "mast3r" / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if summary:
        (f / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if perturbed:
        (f / "mast3r_perturbed").mkdir()
        (f / "mast3r_perturbed" / "phantom_benchmark.json").write_text(json.dumps(perturbed), encoding="utf-8")
    return f


STAT_OK = {"statuses": {"geometry_support": "per-point R/I/U", "lens_calibration": "CALIBRATED", "metric_scale": "NO_SCALE"}}


def test_report_passes_only_when_every_criterion_has_a_receipt(tmp_path: Path) -> None:
    br = _mod("benchmark_report")
    lens = tmp_path / "lens.json"
    lens.write_text(json.dumps({"views": 10, "rms_px": 0.4, "held_out_rms_px": [0.5] * 10, "diversity": {"tilt_deg_max": 20}}), encoding="utf-8")
    sweeps = [
        _sweep(tmp_path, "s1", "straight", _bench(0.1, 0.3, 0.6, arc=150), STAT_OK),
        _sweep(tmp_path, "s2", "twist", _bench(0.12, 0.31, 0.7, arc=260), STAT_OK, perturbed=_bench(0.4, 0.5, 0.9)),
        _sweep(tmp_path, "s3", "repeat-twist", _bench(0.11, 0.3, 0.65, fitted=4.1, arc=250), STAT_OK),
        _sweep(tmp_path, "s4", "low-motion-control", None, None, summary={"status": "NotEnoughMotion: no parallax", **STAT_OK}),
        _sweep(tmp_path, "s5", "high-glare-control", None, {"status": "REFUSED: GLARE (clipped 12%)", **STAT_OK}),
        _sweep(tmp_path, "s6", "wrong-calibration-control", _bench(0.12, 0.31, 0.7), STAT_OK, perturbed=_bench(0.4, 0.5, 0.9)),
    ]
    rep = br.build(sweeps, lens, None, None, tmp_path / "out")
    failing = [c for c in rep["criteria"] if c["pass"] is not True]
    assert not rep["missing"], rep["missing"]
    assert failing == [c for c in failing if c["criterion"].startswith("blind analysis")]  # only the live freeze check may fail here
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "| R | 1000 | 0.1 |" in md and "twist increases supported wall coverage" in md


def test_report_flags_control_that_produced_geometry_and_missing_pieces(tmp_path: Path) -> None:
    br = _mod("benchmark_report")
    sweeps = [
        _sweep(tmp_path, "s1", "straight", _bench(0.3, 0.3, 0.3), STAT_OK),  # no discrimination
        _sweep(tmp_path, "s5", "high-glare-control", _bench(0.1, 0.2, 0.3), STAT_OK),  # control produced trusted geometry
    ]
    rep = br.build(sweeps, None, None, None, tmp_path / "out")
    assert rep["verdict"].startswith("INCOMPLETE OR FAILED")
    crit = {c["criterion"]: c for c in rep["criteria"]}
    assert crit["R<I<U median physical error on every valid sweep"]["pass"] is False
    assert crit["high-glare-control fails explicitly"]["pass"] is False and "PRODUCED GEOMETRY" in crit["high-glare-control fails explicitly"]["evidence"]
    assert any("low-motion-control" in m for m in rep["missing"]) and "lens profile (--lens)" in rep["missing"]


def test_report_wrong_calibration_control_needs_degradation(tmp_path: Path) -> None:
    br = _mod("benchmark_report")
    s = _sweep(tmp_path, "w", "wrong-calibration-control", _bench(0.12, 0.31, 0.7), STAT_OK, perturbed=_bench(0.12, 0.31, 0.7))
    text, ok = br.control_disposition(br.gather(s))
    assert ok is False and "NO DEGRADATION" in text
