# UNFILED-IP
"""Phantom benchmark: does R identify physically accurate geometry, and what mm error does each class predict?

    uv run python tools/phantom_benchmark.py <graded.ply or mast3r_evidence.ply> --diameter-mm 8.0 [--lens lens.json --scale-mm X]

Scan a straight tube of known inner diameter. The script fits a cylinder axis to the R points
(robust: PCA of the R cloud, then median radius), then reports per-class radial error against the
known radius. With no metric scale, the fitted R radius itself sets the scale (then only RELATIVE
error per class is meaningful and it is labelled so). With --scale-mm (a known distance, e.g. an
insertion mark moved between two sweeps) the error is absolute millimetres.
Pre-registered expectation: median error orders R < I < U. EXPERIMENTAL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CLASS_RGB: dict[tuple[int, ...], str] = {(37, 99, 235): "R", (245, 158, 11): "I", (120, 120, 120): "U", (170, 170, 170): "U"}


def fit_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    return c, vt[0]


def radial_distance(points: np.ndarray, c: np.ndarray, axis: np.ndarray) -> np.ndarray:
    d = points - c
    along = d @ axis
    return np.asarray(np.linalg.norm(d - np.outer(along, axis), axis=1), dtype=np.float64)


def slice_radius_profile(points: np.ndarray, c: np.ndarray, axis: np.ndarray, slice_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Median radius of the R wall per axial slice: tolerates narrowing / bumps a single cylinder cannot."""
    along = (points - c) @ axis
    rad = radial_distance(points, c, axis)
    edges = np.arange(along.min(), along.max() + slice_mm, slice_mm)
    centres, radii = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (along >= lo) & (along < hi)
        if m.sum() >= 20:
            centres.append(0.5 * (lo + hi))
            radii.append(float(np.median(rad[m])))
    return np.array(centres), np.array(radii)


def local_radius(points: np.ndarray, c: np.ndarray, axis: np.ndarray, prof_z: np.ndarray, prof_r: np.ndarray) -> np.ndarray:
    along = (points - c) @ axis
    return np.asarray(np.interp(along, prof_z, prof_r), dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    ap.add_argument("--diameter-mm", type=float, required=True)
    ap.add_argument("--scale-mm", type=float, default=None, help="known metric length to set the scale (units -> mm)")
    ap.add_argument("--scale-units", type=float, default=None, help="that length measured in cloud units")
    ap.add_argument("--profile", action="store_true", help="compare each class against the R wall's own axial radius "
                    "profile instead of a single known radius (for canals with narrowing/bumps)")
    ap.add_argument("--slice-mm", type=float, default=1.0)
    a = ap.parse_args()
    from eczis.lab.grade_dense import read_ply

    xyz, rgb = read_ply(a.ply)
    if rgb is None:
        raise SystemExit("PLY has no colours; need an evidence-coloured cloud")
    cls = np.array([CLASS_RGB.get(tuple(int(v) for v in c), "U") for c in rgb])
    r_pts = xyz[cls == "R"]
    if len(r_pts) < 100:
        raise SystemExit(f"only {len(r_pts)} R points; benchmark needs a supported wall")
    c, axis = fit_axis(r_pts)
    rad_r = radial_distance(r_pts, c, axis)
    true_r = a.diameter_mm / 2.0
    if a.scale_mm and a.scale_units:
        scale = a.scale_mm / a.scale_units
        mode = "absolute (metric scale supplied)"
    else:
        scale = true_r / float(np.median(rad_r))
        mode = "RELATIVE: scale set from the R radius itself, so R error is zero by construction; compare I and U to R"
    out = {"mode": mode, "scale_units_to_mm": round(scale, 6), "true_radius_mm": true_r, "axis_from": "PCA of R points",
           "reference": "R-wall axial radius profile" if a.profile else "single known radius", "classes": {}}
    prof_z, prof_r = slice_radius_profile(r_pts * scale, c * scale, axis, a.slice_mm) if a.profile else (None, None)
    for k in ("R", "I", "U"):
        p = xyz[cls == k]
        if len(p) == 0:
            continue
        ref = local_radius(p * scale, c * scale, axis, prof_z, prof_r) if (a.profile and prof_z is not None and prof_r is not None) else true_r
        e = np.abs(radial_distance(p, c, axis) * scale - ref)
        out["classes"][k] = {"n": int(len(p)), "median_abs_err_mm": round(float(np.median(e)), 3),
                             "p90_abs_err_mm": round(float(np.percentile(e, 90)), 3), "frac_within_0.5mm": round(float(np.mean(e < 0.5)), 3)}
    meds = [out["classes"][k]["median_abs_err_mm"] for k in ("R", "I", "U") if k in out["classes"]]
    out["ordering_R<I<U"] = bool(all(meds[i] <= meds[i + 1] for i in range(len(meds) - 1)))
    out["notes"] = ["straight axis from PCA; --profile tolerates narrowing/bumps along it but not a bent canal",
                    "with --profile, R error is relative to R's own median per slice (expect small by construction); the "
                    "test is whether I and U are worse",
                    "measurement record, not a diagnosis"]
    print(json.dumps(out, indent=2))
    Path(a.ply).with_name("phantom_benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if out["ordering_R<I<U"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
