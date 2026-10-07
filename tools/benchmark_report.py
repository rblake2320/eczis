# UNFILED-IP
"""Assemble the physical phantom benchmark record from the sweep folders: the decisive table, the controls, the hashes.

    uv run python tools/benchmark_report.py artifacts/earscan/<straight> artifacts/earscan/<twist> ... \\
        --lens artifacts/calib/lens_camera1.json --out artifacts/benchmark/<date>

Each sweep folder must carry phantom_record.json (tools/phantom_record.py: role + tube measurements) and,
for valid sweeps, mast3r/phantom_benchmark.json (tools/phantom_benchmark.py) and mast3r/result.json; for
controls, whatever the pipeline wrote (a refusal in result.json/summary.json, or a perturbed grading).
Nothing is recomputed here: the report binds what exists, judges each pre-registered criterion, and
lists what is missing. A missing receipt is reported as missing, never filled in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VALID_ROLES = ("straight", "twist", "repeat-twist")
CONTROL_ROLES = ("low-motion-control", "high-glare-control", "wrong-calibration-control")
REPEAT_DIAMETER_TOL_MM = 0.3
LENS_STATUSES = {"CALIBRATED", "UNCALIBRATED"}
SCALE_STATUSES = {"NO_SCALE", "SCALED"}


def _load(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def gather(folder: Path) -> dict[str, Any]:
    rec = _load(folder / "phantom_record.json")
    m = folder / "mast3r"
    bench = _load(m / "phantom_benchmark.json") or _load(folder / "phantom_benchmark.json")
    result = _load(m / "result.json")
    summary = _load(folder / "summary.json")
    perturbed = _load(folder / "mast3r_perturbed" / "phantom_benchmark.json")
    pres = _load(folder / "mast3r_perturbed" / "result.json")
    return {"folder": str(folder), "record": rec, "benchmark": bench, "result": result, "summary": summary,
            "scale": _load(folder / "scale.json"), "perturbed_benchmark": perturbed, "perturbed_result": pres,
            "hashes": {k: _sha(folder / v) for k, v in (("frames.npz", "frames.npz"), ("graded.ply", "mast3r/graded.ply"),
                                                         ("natural.ply", "mast3r/mast3r_natural.ply"),
                                                         ("evidence.ply", "mast3r/mast3r_evidence.ply"))}}


def control_disposition(s: dict[str, Any]) -> tuple[str, bool]:
    """Did a failure control fail explicitly? (refusal in result/summary, or no geometry produced)."""
    role = (s["record"] or {}).get("role", "")
    res, summ = s["result"] or {}, s["summary"] or {}
    status = str(res.get("status", "")) + " " + str(summ.get("status", "")) + " " + str(summ.get("error", ""))
    if role == "wrong-calibration-control":
        b, pb = s["benchmark"], s["perturbed_benchmark"]
        if not (b and pb):
            return "MISSING: needs both the valid benchmark and mast3r_perturbed/phantom_benchmark.json", False
        r_ok, r_bad = b["classes"].get("R", {}), pb["classes"].get("R", {})
        worse = r_bad.get("median_abs_err_mm", 0) > r_ok.get("median_abs_err_mm", 0)
        weaker = r_bad.get("n", 0) < r_ok.get("n", 0)
        return (f"R median {r_ok.get('median_abs_err_mm')} -> {r_bad.get('median_abs_err_mm')} mm, R points {r_ok.get('n')} -> {r_bad.get('n')}: "
                + ("errors increased or support weakened (expected)" if worse or weaker else "NO DEGRADATION: calibration has no grip"), worse or weaker)
    if "REFUSED" in status or "NotEnoughMotion" in status or "Not enough" in status:
        return f"refused explicitly: {status.strip()[:160]}", True
    if s["benchmark"] or (s["hashes"].get("graded.ply")) or (folder_has_ply(s)):
        return "PRODUCED GEOMETRY: control did not fail (evidence policy needs repair)", False
    return "no output recorded (neither refusal nor geometry): incomplete", False


def folder_has_ply(s: dict[str, Any]) -> bool:
    return (Path(s["folder"]) / "scan.ply").exists()


def statuses_independent(s: dict[str, Any]) -> tuple[str, bool]:
    st = ((s["result"] or {}).get("statuses") or (s["summary"] or {}).get("statuses") or {})
    if not st:
        return "no statuses block", False
    lens, scale = str(st.get("lens_calibration", "")), str(st.get("metric_scale", ""))
    ok = (lens in LENS_STATUSES or lens.startswith("PERTURBED_CONTROL")) and scale in SCALE_STATUSES and "geometry_support" in st
    return f"geometry={st.get('geometry_support')!r} lens={lens!r} scale={scale!r}", ok


def build(sweeps: list[Path], lens: Path | None, print_record: Path | None, freeze: Path | None, out: Path) -> dict[str, Any]:
    from eczis.lab.policy_freeze import check

    gathered = [gather(f) for f in sweeps]
    by_role: dict[str, list[dict[str, Any]]] = {}
    for g in gathered:
        by_role.setdefault((g["record"] or {}).get("role", "UNRECORDED"), []).append(g)
    tables, criteria, missing = [], [], []
    for g in gathered:
        role = (g["record"] or {}).get("role", "UNRECORDED")
        if role == "UNRECORDED":
            missing.append(f"{g['folder']}: no phantom_record.json")
        b = g["benchmark"]
        if role in VALID_ROLES:
            if not b:
                missing.append(f"{g['folder']} ({role}): no phantom_benchmark.json")
                continue
            tables.append({"folder": Path(g["folder"]).name, "role": role, "mode": b.get("mode"), "axis_from": b.get("axis_from"),
                           "fitted_diameter_mm": round(2 * float(b.get("fitted_R_radius_mm", 0)), 3),
                           "true_diameter_mm": 2 * float(b.get("true_radius_mm", 0)), "R_arc_coverage_deg": b.get("R_arc_coverage_deg"),
                           "ordering_median": b.get("ordering_R<I<U"), "ordering_p90": b.get("ordering_R<I<U_p90"),
                           "ordering_p95": b.get("ordering_R<I<U_p95"), "blind": (b.get("blind_analysis") or {}).get("status"),
                           "profile_vs_global_mm": b.get("profile_vs_global_disagreement_mm"),
                           "slices": b.get("diameter_error_by_slice"), "classes": b["classes"]})
    valid = [t for t in tables]
    # criteria, each pre-registered in docs/MICROSCOPE-LAB.md
    lp = _load(lens) if lens else None
    if lp:
        ho = lp.get("held_out_rms_px") or []
        criteria.append({"criterion": "calibration >= 8 diverse views, held-out stable", "pass": bool(lp.get("views", 0) >= 8 and ho and
                         max(ho) <= max(1.5, 2 * float(lp.get("rms_px", 9)))),
                         "evidence": f"views {lp.get('views')}, rms {lp.get('rms_px')}, held-out max {max(ho) if ho else None}, diversity {lp.get('diversity')}"})
    else:
        missing.append("lens profile (--lens)")
    absolute = [t for t in valid if str(t["mode"]).startswith("absolute")]
    if absolute:
        errs = [abs(t["fitted_diameter_mm"] - t["true_diameter_mm"]) for t in absolute]
        criteria.append({"criterion": "known scale recovered (fitted diameter vs measured, absolute mode)", "pass": bool(max(errs) <= REPEAT_DIAMETER_TOL_MM),
                         "evidence": [f"{t['folder']}: fitted {t['fitted_diameter_mm']} vs {t['true_diameter_mm']} mm" for t in absolute]})
    else:
        criteria.append({"criterion": "known scale recovered", "pass": None, "evidence": "no sweep benchmarked in absolute mode (needs --scale-mm/--scale-units)"})
    for key, label in (("ordering_median", "R<I<U median physical error on every valid sweep"), ("ordering_p90", "R<I<U at P90"), ("ordering_p95", "R<I<U at P95")):
        vals = [t[key] for t in valid]
        criteria.append({"criterion": label, "pass": bool(vals and all(vals)), "evidence": {t["folder"]: t[key] for t in valid}})
    cov = {t["role"]: t["R_arc_coverage_deg"] for t in valid if t["role"] in ("straight", "twist")}
    if "straight" in cov and "twist" in cov:
        criteria.append({"criterion": "twist increases supported wall coverage", "pass": bool(cov["twist"] > cov["straight"]), "evidence": cov})
    reps = [t for t in valid if t["role"] in ("twist", "repeat-twist")]
    if len(reps) >= 2:
        d = abs(reps[0]["fitted_diameter_mm"] - reps[1]["fitted_diameter_mm"])
        criteria.append({"criterion": "repeated sweeps give compatible profiles", "pass": bool(d <= REPEAT_DIAMETER_TOL_MM),
                         "evidence": f"fitted diameters {reps[0]['fitted_diameter_mm']} vs {reps[1]['fitted_diameter_mm']} mm (|d| {d:.3f})"})
    for role in CONTROL_ROLES:
        for g in by_role.get(role, []):
            text, ok = control_disposition(g)
            criteria.append({"criterion": f"{role} fails explicitly", "pass": ok, "evidence": f"{Path(g['folder']).name}: {text}"})
        if role not in by_role:
            missing.append(f"control sweep: {role}")
    indep = [statuses_independent(g) for g in gathered if g["result"] or g["summary"]]
    criteria.append({"criterion": "lens calibration and metric scale remain independent statuses", "pass": bool(indep and all(ok for _, ok in indep)),
                     "evidence": [t for t, _ in indep]})
    blind = [t["blind"] for t in valid]
    criteria.append({"criterion": "blind analysis (policy frozen before truth entered)", "pass": bool(blind and all(b == "BLIND" for b in blind)),
                     "evidence": {t["folder"]: t["blind"] for t in valid} | {"now": check(freeze) if freeze else check()}})
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": commit, "sweeps": gathered, "tables": tables,
              "criteria": criteria, "missing": missing, "lens": lp, "lens_sha256": _sha(lens) if lens else None,
              "print_record": _load(print_record) if print_record else None,
              "verdict": ("ALL PRE-REGISTERED CRITERIA PASS" if criteria and all(c["pass"] for c in criteria) and not missing
                          else "INCOMPLETE OR FAILED: see criteria and missing"),
              "claim_if_pass": "On a physically measured phantom, the evidence classes stratified real 3D reconstruction error while "
                               "acquisitions lacking motion or usable image quality were refused.",
              "measurement_record_not_diagnosis": True}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def markdown(rep: dict[str, Any]) -> str:
    lines = [f"# Phantom benchmark record — {rep['generated_at']} (commit {rep['commit'][:9]})", "", f"**Verdict:** {rep['verdict']}", ""]
    for t in rep["tables"]:
        lines += [f"## {t['folder']} — {t['role']} ({t['mode']}; axis: {t['axis_from']}; blind: {t['blind']})", "",
                  f"fitted diameter {t['fitted_diameter_mm']} mm vs measured {t['true_diameter_mm']} mm; R arc {t['R_arc_coverage_deg']} deg; "
                  f"profile-vs-global {t['profile_vs_global_mm']} mm", "",
                  "| Class | Points | Median error (mm) | P90 (mm) | P95 (mm) | Within 0.5 mm |", "|---|---:|---:|---:|---:|---:|"]
        for k in ("R", "I", "U"):
            c = t["classes"].get(k)
            if c:
                lines.append(f"| {k} | {c['n']} | {c['median_abs_err_mm']} | {c['p90_abs_err_mm']} | {c['p95_abs_err_mm']} | {c['frac_within_0.5mm']} |")
        lines.append("")
        if t.get("slices"):
            lines += ["| z (mm) | R diameter (mm) | error (mm) |", "|---:|---:|---:|"]
            lines += [f"| {s['z_mm']} | {s['R_diameter_mm']} | {s['error_mm']} |" for s in t["slices"]]
            lines.append("")
    lines += ["## Pre-registered criteria", "", "| Criterion | Pass | Evidence |", "|---|---|---|"]
    for c in rep["criteria"]:
        lines.append(f"| {c['criterion']} | {c['pass']} | {json.dumps(c['evidence'], default=str)[:300]} |")
    if rep["missing"]:
        lines += ["", "## Missing", ""] + [f"- {m}" for m in rep["missing"]]
    lines += ["", "## Bindings", ""]
    for s in rep["sweeps"]:
        lines.append(f"- {s['folder']}: " + ", ".join(f"{k} {v[:12] if v else 'absent'}" for k, v in s["hashes"].items())
                     + (f"; env {s['result'].get('environment')}" if s.get("result") and s["result"].get("environment") else ""))
    lines += [f"- lens sha256 {rep['lens_sha256']}", "", "Measurement record, not a diagnosis."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", type=Path, nargs="+")
    ap.add_argument("--lens", type=Path, default=None)
    ap.add_argument("--print-record", type=Path, default=None)
    ap.add_argument("--freeze", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("artifacts/benchmark") / time.strftime("report-%Y%m%d-%H%M%S"))
    a = ap.parse_args()
    rep = build(a.sweeps, a.lens, a.print_record, a.freeze, a.out)
    print((a.out / "report.md").read_text(encoding="utf-8"))
    return 0 if rep["verdict"].startswith("ALL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
