# UNFILED-IP
"""Phantom benchmark: does R identify physically accurate geometry, and what mm error does each class carry?

    uv run python tools/phantom_benchmark.py <graded.ply or mast3r_evidence.ply> --diameter-mm 8.0 \\
        --trajectory <folder>/mast3r/<name>.txt --scale-mm 5.0 --scale-units 0.45 [--profile]

Scan a straight tube of known inner diameter. The axis is taken from the CAMERA TRAJECTORY
(`--trajectory`, TUM file): the probe moves along the tube, so the camera path is the axis
direction; the axis position and the radius then come from a circle fit of the R points in the
plane perpendicular to it. Without a trajectory the axis falls back to PCA of the R cloud, which
is only valid when the scanned patch is longer than it is wide AND wraps most of the tube; the
tool measures both and REFUSES (exit 2) when the axis is underdetermined instead of reporting a
number against a wrong axis.

Every class is scored against the KNOWN radius (physical error, needs --scale-mm/--scale-units
for absolute mm; otherwise the fitted R radius sets the scale and the error is relative).
`--profile` adds a second column: error against the R wall's own per-slice median radius, a
self-consistency figure that tolerates narrowing/bumps but is NOT a physical error.
Pre-registered expectation: median physical error strictly orders R < I < U. EXPERIMENTAL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CLASS_RGB: dict[tuple[int, ...], str] = {(37, 99, 235): "R", (245, 158, 11): "I", (120, 120, 120): "U", (170, 170, 170): "U"}
MIN_ARC_DEG = 90.0  # a circle fit on a shorter arc is unstable
MAX_RESIDUAL_ANY = 0.15  # cylinder-consistency sanity for any axis source
MIN_SINGULAR_RATIO = 1.5  # PCA axis is only trusted when the axial spread clearly dominates the cross-section
# (a full tube ~3 diameters long gives ~2.3; a patch as long as it is wide gives ~1.0 and PCA picks a chord)
MIN_WRAP_DEG = 170.0  # ... and the R wall wraps about half the tube (a clean half-wrap estimates ~176)
MAX_RESIDUAL_PCA = 0.08  # ... and the R points lie on a circle about that axis (a chord axis gives ~0.2+)


def refuse(message: str) -> None:
    """Exit 2: the benchmark declined to produce a number (distinct from 1 = measured, ordering failed)."""
    print(message, file=sys.stderr)
    raise SystemExit(2)


def principal_direction(points: np.ndarray) -> np.ndarray:
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    return np.asarray(vt[0], dtype=np.float64)


def perpendicular_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def fit_circle_about_axis(points: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares circle (Kasa) of the points projected on the plane perpendicular to `axis`.

    Returns (a point on the axis, radius, arc coverage in degrees). Works on a partial wall.
    """
    e1, e2 = perpendicular_frame(axis)
    x, y = points @ e1, points @ e2
    a_mat = np.stack([x, y, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(a_mat, -(x * x + y * y), rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r = float(np.sqrt(max(cx * cx + cy * cy - sol[2], 1e-12)))
    centre = cx * e1 + cy * e2 + float(np.mean(points @ axis)) * axis
    return centre, r, arc_from_cross_section(np.stack([x, y], axis=1))


def arc_from_cross_section(xy: np.ndarray) -> float:
    """Angular extent (deg) of a wall arc from its chord/sagitta shape, independent of the circle fit.

    A circular arc of angle t has minor/major extent (1 - cos(t/2)) / (2 sin(t/2)) for t <= 180 deg and
    (1 - cos(t/2)) / 2 beyond; both are monotonic, so the ratio of robust extents along the 2D
    principal axes inverts to t by bisection. The fit's own centre is not used (a bad fit on a
    small arc would otherwise report a generous span).
    """
    c = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - c, full_matrices=False)
    proj = (xy - c) @ vt.T
    ext = np.percentile(proj, 98, axis=0) - np.percentile(proj, 2, axis=0)
    ratio = float(ext[1] / max(ext[0], 1e-12))

    def f(t_deg: float) -> float:
        h = np.radians(t_deg / 2)
        return float((1 - np.cos(h)) / (2 * np.sin(h))) if t_deg <= 180 else float((1 - np.cos(h)) / 2)

    lo, hi = 1.0, 360.0
    if ratio >= f(hi):
        return 360.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < ratio else (lo, mid)
    return round(0.5 * (lo + hi), 1)


def radial_distance(points: np.ndarray, c: np.ndarray, axis: np.ndarray) -> np.ndarray:
    d = points - c
    along = d @ axis
    return np.asarray(np.linalg.norm(d - np.outer(along, axis), axis=1), dtype=np.float64)


def slice_radius_profile(points: np.ndarray, c: np.ndarray, axis: np.ndarray, slice_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Median radius of the R wall per axial slice (self-consistency reference, tolerates narrowing)."""
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


def choose_axis(r_pts: np.ndarray, trajectory: Path | None) -> tuple[np.ndarray, str, dict[str, float]]:
    """Axis direction from the camera path when given; otherwise PCA, gated on patch shape."""
    pca = principal_direction(r_pts)
    if trajectory is not None:
        from eczis.lab.grade_dense import read_tum_trajectory

        poses = read_tum_trajectory(trajectory)
        if len(poses) < 2:
            raise SystemExit(f"{trajectory}: fewer than 2 poses; cannot take an axis from it")
        cam = np.array([p.position for p in poses])
        axis = principal_direction(cam)
        if cam[-1] @ axis < cam[0] @ axis:
            axis = -axis
        return axis, "camera trajectory (principal direction of the camera path)", {
            "cos_to_pca_axis": round(abs(float(axis @ pca)), 3), "camera_path_units": round(float(np.linalg.norm(cam[-1] - cam[0])), 4)}
    c, r_fit, arc = fit_circle_about_axis(r_pts, pca)
    s = np.linalg.svd(r_pts - r_pts.mean(axis=0), compute_uv=False)
    resid = float(np.median(np.abs(radial_distance(r_pts, c, pca) - r_fit)) / max(r_fit, 1e-12))
    shape = {"singular_ratio_axial_over_transverse": round(float(s[0] / max(s[1], 1e-12)), 3), "arc_coverage_deg": arc,
             "radial_residual_over_radius": round(resid, 3)}
    if shape["singular_ratio_axial_over_transverse"] < MIN_SINGULAR_RATIO or arc < MIN_WRAP_DEG or resid > MAX_RESIDUAL_PCA:
        refuse("axis underdetermined for PCA: axial/transverse spread ratio "
               f"{shape['singular_ratio_axial_over_transverse']:.2f} (need >= {MIN_SINGULAR_RATIO}), R wraps {arc:.0f} deg "
               f"(need >= {MIN_WRAP_DEG:.0f}), radial residual {resid:.2f} of radius (need <= {MAX_RESIDUAL_PCA}); "
               "pass --trajectory <tum.txt> so the camera path sets the axis")
    return pca, "PCA of R points (patch long and wrapped enough)", shape


def benchmark(ply: Path, diameter_mm: float, trajectory: Path | None, scale_mm: float | None, scale_units: float | None,
              profile: bool, slice_mm: float) -> dict[str, object]:
    from eczis.lab.grade_dense import read_ply

    xyz, rgb = read_ply(ply)
    if rgb is None:
        raise SystemExit("PLY has no colours; need an evidence-coloured cloud")
    cls = np.array([CLASS_RGB.get(tuple(int(v) for v in c), "U") for c in rgb])
    r_pts = xyz[cls == "R"]
    if len(r_pts) < 100:
        raise SystemExit(f"only {len(r_pts)} R points; benchmark needs a supported wall")
    axis, axis_source, axis_diag = choose_axis(r_pts, trajectory)
    c, r_fit, arc = fit_circle_about_axis(r_pts, axis)
    if arc < MIN_ARC_DEG:
        refuse(f"R wall covers only {arc:.0f} deg of the tube (need >= {MIN_ARC_DEG:.0f}); the circle fit is unstable")
    resid = float(np.median(np.abs(radial_distance(r_pts, c, axis) - r_fit)) / max(r_fit, 1e-12))
    if resid > MAX_RESIDUAL_ANY:
        refuse(f"R points do not lie on a cylinder about the chosen axis (median radial residual {resid:.2f} of the radius, "
               f"need <= {MAX_RESIDUAL_ANY}); the trajectory may not follow the tube, or the cloud is not a tube")
    true_r = diameter_mm / 2.0
    if scale_mm and scale_units:
        scale = scale_mm / scale_units
        mode = "absolute (metric scale supplied)"
    else:
        scale = true_r / r_fit
        mode = "RELATIVE: scale set from the fitted R radius itself, so R's physical error is near zero by construction; compare I and U to R"
    out: dict[str, object] = {"mode": mode, "scale_units_to_mm": round(scale, 6), "true_radius_mm": true_r,
                              "fitted_R_radius_mm": round(r_fit * scale, 3), "axis_from": axis_source, "axis_diagnostics": axis_diag,
                              "R_arc_coverage_deg": arc, "R_radial_residual_over_radius": round(resid, 3), "classes": {}}
    prof_z, prof_r = slice_radius_profile(r_pts * scale, c * scale, axis, slice_mm) if profile else (None, None)
    classes: dict[str, dict[str, float | int]] = {}
    for k in ("R", "I", "U"):
        p = xyz[cls == k]
        if len(p) == 0:
            continue
        rad_mm = radial_distance(p, c, axis) * scale
        e = np.abs(rad_mm - true_r)
        row: dict[str, float | int] = {"n": int(len(p)), "median_abs_err_mm": round(float(np.median(e)), 3),
                                       "p90_abs_err_mm": round(float(np.percentile(e, 90)), 3),
                                       "p95_abs_err_mm": round(float(np.percentile(e, 95)), 3),
                                       "frac_within_0.5mm": round(float(np.mean(e < 0.5)), 3)}
        if profile and prof_z is not None and prof_r is not None and len(prof_z):
            e2 = np.abs(rad_mm - local_radius(p * scale, c * scale, axis, prof_z, prof_r))
            row["median_err_vs_R_profile_mm"] = round(float(np.median(e2)), 3)
        classes[k] = row
    out["classes"] = classes
    meds = [classes[k]["median_abs_err_mm"] for k in ("R", "I", "U") if k in classes]
    out["ordering_R<I<U"] = bool(len(meds) == 3 and meds[0] < meds[1] < meds[2])
    for q in ("p90_abs_err_mm", "p95_abs_err_mm"):
        v = [classes[k][q] for k in ("R", "I", "U") if k in classes]
        out[f"ordering_R<I<U_{q.split('_')[0]}"] = bool(len(v) == 3 and v[0] < v[1] < v[2])
    if profile and prof_z is not None and prof_r is not None and len(prof_z):
        # diameter error by axial slice (R wall vs the known diameter) and global-vs-profile disagreement
        out["diameter_error_by_slice"] = [{"z_mm": round(float(z), 2), "R_diameter_mm": round(float(2 * r), 3),
                                           "error_mm": round(float(2 * r - diameter_mm), 3)} for z, r in zip(prof_z, prof_r, strict=True)]
        out["profile_vs_global_disagreement_mm"] = round(float(np.median(np.abs(2 * prof_r - 2 * r_fit * scale))), 3)
        out["slice_error_p90_mm"] = round(float(np.percentile(np.abs(2 * prof_r - diameter_mm), 90)), 3)
    from eczis.lab.policy_freeze import check

    out["blind_analysis"] = check()
    out["notes"] = ["physical error = |radius - known radius| per point; ordering is on the physical median, strict",
                    "ordering_*_p90/p95 report the same test on the tails",
                    "blind_analysis BLIND means the grading policy was frozen before this truth was entered and is unchanged",
                    "the R-profile column (with --profile) is self-consistency, not physical error",
                    "axis from the camera path when --trajectory is given; PCA only when the patch is long and wrapped enough",
                    "measurement record, not a diagnosis"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    ap.add_argument("--diameter-mm", type=float, required=True)
    ap.add_argument("--trajectory", type=Path, default=None, help="TUM trajectory of the sweep; sets the tube axis")
    ap.add_argument("--scale-mm", type=float, default=None, help="known metric length to set the scale (units -> mm)")
    ap.add_argument("--scale-units", type=float, default=None, help="that length measured in cloud units")
    ap.add_argument("--profile", action="store_true", help="also report error against the R wall's own per-slice radius")
    ap.add_argument("--slice-mm", type=float, default=1.0)
    a = ap.parse_args()
    out = benchmark(a.ply, a.diameter_mm, a.trajectory, a.scale_mm, a.scale_units, a.profile, a.slice_mm)
    print(json.dumps(out, indent=2))
    Path(a.ply).with_name("phantom_benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if out["ordering_R<I<U"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
