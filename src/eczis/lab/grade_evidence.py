# UNFILED-IP
"""Evidence grading of a dense reconstruction from per-keyframe exports (MASt3R-SLAM evidence.npz).

The reconstructor is treated as a proposal to be cross-checked, not as ground truth. The checks
are only PARTLY independent of it: the depth test compares the reconstructor's own per-keyframe
depths with each other, and a calibration-free run fits K from its own point map; the colour
agreement and the parallax rule are the parts that do not come from the reconstructor. For every
keyframe pixel with a 3D point, ECZIS checks it against the OTHER keyframes:

  reprojection  project the world point into keyframe j; it must land in the image, in front
  depth         keyframe j's own point at that pixel must agree in depth (relative tolerance)
  colour        keyframe j's image at that pixel must agree in intensity (absolute tolerance)
  parallax      viewing directions from the agreeing keyframes must span >= an angle

A point is R only if >= 2 OTHER keyframes agree on depth and colour and the parallax rule holds;
I if exactly one other keyframe agrees; U otherwise. Hard rules override the counts:
one source view -> never R; a conflicting depth (another keyframe sees a closer surface at that
pixel, i.e. a drifted sheet) -> U; a point from a keyframe after a tracking loss -> at most I.
The reconstructor's own confidence is reported but never used to promote.

Scale and lens remain independent statuses (NO_SCALE / UNCALIBRATED) and are never inferred.
EXPERIMENTAL; never changes production M/R/I/U.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
F32 = npt.NDArray[np.float32]

POLICY = {
    "version": "grade_evidence/0.1",
    "depth_rel_tol": 0.08,  # |d_j - d_pred| / d_pred
    "conflict_rel": 0.15,  # keyframe j sees a surface this much CLOSER: occluder or drifted sheet
    "colour_tol": 0.12,  # intensity difference on a 0..1 scale
    "min_parallax_deg": 2.0,
    "r_min_agree": 2,  # other keyframes agreeing
    "i_min_agree": 1,
}


@dataclass(frozen=True, slots=True)
class Keyframe:
    index: int
    frame_id: int
    t: float
    rot_wc: F64
    pos: F64
    scale: float
    points_w: F64  # (h, w, 3) world coordinates
    conf: F32
    gray: F64  # (h, w) 0..1
    k: F64 | None


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> F64:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def load_evidence(path: str | Path, k_fallback: F64 | None = None) -> list[Keyframe]:
    d = np.load(path)
    n = int(d["n_keyframes"])
    out = []
    for i in range(n):
        twc = np.asarray(d[f"kf{i}_T_WC"], dtype=np.float64).ravel()
        x, y, z, qx, qy, qz, qw = twc[:7]
        s = float(twc[7]) if twc.size >= 8 else 1.0
        rot = _quat_to_rot(qx, qy, qz, qw)
        pos = np.array([x, y, z])
        xc = np.asarray(d[f"kf{i}_X_canon"], dtype=np.float64)
        pw = (xc.reshape(-1, 3) * s) @ rot.T + pos  # Sim3 act: scale, rotate, translate
        img = np.asarray(d[f"kf{i}_img"], dtype=np.float64) / 255.0
        gray = img @ np.array([0.299, 0.587, 0.114]) if img.ndim == 3 else img
        k = np.asarray(d[f"kf{i}_K"], dtype=np.float64) if f"kf{i}_K" in d else k_fallback
        out.append(Keyframe(i, int(d[f"kf{i}_frame_id"]), float(d[f"kf{i}_t"]), rot, pos, s,
                            pw.reshape(xc.shape), np.asarray(d[f"kf{i}_conf"], dtype=np.float32), gray, k))
    return out


def _intrinsics_from_points(kf: Keyframe) -> F64:
    """Calibration-free runs carry no K: fit fx, fy, cx, cy from the keyframe's own point map."""
    h, w = kf.gray.shape
    pc = (kf.points_w.reshape(-1, 3) - kf.pos) @ kf.rot_wc  # world -> camera
    z = pc[:, 2]
    ok = z > 1e-6
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    uu, vv = u.ravel()[ok], v.ravel()[ok]
    xn, yn = pc[ok, 0] / z[ok], pc[ok, 1] / z[ok]
    ax = np.polyfit(xn, uu, 1)
    ay = np.polyfit(yn, vv, 1)
    return np.array([[ax[0], 0, ax[1]], [0, ay[0], ay[1]], [0, 0, 1.0]])


@dataclass
class Grade:
    classes: npt.NDArray[Any]  # per point "R"/"I"/"U"
    agree: npt.NDArray[np.int32]
    conflict: npt.NDArray[np.int32]
    parallax_deg: F64
    reasons: dict[str, int] = field(default_factory=dict)


def grade_keyframes(kfs: list[Keyframe], loss_after_frame: int | None = None, policy: dict[str, Any] = POLICY,
                    stride: int = 1) -> tuple[F64, npt.NDArray[np.uint8], Grade]:
    """Grade every keyframe pixel (subsampled by stride) against all other keyframes."""
    ks = [kf.k if kf.k is not None else _intrinsics_from_points(kf) for kf in kfs]
    pts_all, col_all, cls_all, agr_all, conf_all, par_all, cfl_all = [], [], [], [], [], [], []
    reasons = {"no_agreeing_view": 0, "depth_conflict_sheet": 0, "post_loss_capped": 0, "parallax_too_small": 0}
    for i, kf in enumerate(kfs):
        h, w = kf.gray.shape
        pw = kf.points_w[::stride, ::stride].reshape(-1, 3)
        src_gray = kf.gray[::stride, ::stride].ravel()
        n = pw.shape[0]
        agree = np.zeros(n, dtype=np.int32)
        conflict = np.zeros(n, dtype=np.int32)
        dirs_first = (pw - kf.pos)
        dirs_first /= np.maximum(np.linalg.norm(dirs_first, axis=1, keepdims=True), 1e-12)
        min_cos = np.ones(n)
        for j, other in enumerate(kfs):
            if j == i:
                continue
            kj = ks[j]
            pc = (pw - other.pos) @ other.rot_wc  # world -> camera j
            z = pc[:, 2]
            front = z > 1e-6
            u = kj[0, 0] * pc[:, 0] / np.where(front, z, 1.0) + kj[0, 2]
            v = kj[1, 1] * pc[:, 1] / np.where(front, z, 1.0) + kj[1, 2]
            hj, wj = other.gray.shape
            inside = front & (u >= 0) & (u < wj) & (v >= 0) & (v < hj)
            if not inside.any():
                continue
            ui, vi = u[inside].astype(int), v[inside].astype(int)
            # keyframe j's own depth at that pixel
            own = other.points_w[vi, ui]
            own_z = ((own - other.pos) @ other.rot_wc)[:, 2]
            pred_z = z[inside]
            rel = (own_z - pred_z) / np.maximum(pred_z, 1e-9)
            depth_ok = np.abs(rel) <= policy["depth_rel_tol"]
            closer = rel < -policy["conflict_rel"]  # j sees a nearer surface where our point projects
            col_ok = np.abs(other.gray[vi, ui] - src_gray[inside]) <= policy["colour_tol"]
            idx = np.flatnonzero(inside)
            good = idx[depth_ok & col_ok]
            agree[good] += 1
            conflict[idx[closer]] += 1
            d = pw[good] - other.pos
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            min_cos[good] = np.minimum(min_cos[good], np.einsum("ij,ij->i", d, dirs_first[good]))
        par = np.degrees(np.arccos(np.clip(min_cos, -1, 1)))
        cls = np.full(n, "U", dtype="<U1")
        cls[agree >= policy["i_min_agree"]] = "I"
        r_ok = (agree >= policy["r_min_agree"]) & (par >= policy["min_parallax_deg"])
        reasons["parallax_too_small"] += int(((agree >= policy["r_min_agree"]) & ~r_ok).sum())
        cls[r_ok] = "R"
        cls[conflict > 0] = "U"
        reasons["depth_conflict_sheet"] += int((conflict > 0).sum())
        reasons["no_agreeing_view"] += int((agree == 0).sum())  # no other keyframe agreed: unseen OR seen but depth/colour failed
        if loss_after_frame is not None and kf.frame_id > loss_after_frame:
            capped = cls == "R"
            reasons["post_loss_capped"] += int(capped.sum())
            cls[capped] = "I"
        pts_all.append(pw)
        col_all.append(np.repeat((src_gray * 255).astype(np.uint8)[:, None], 3, axis=1))
        cls_all.append(cls)
        agr_all.append(agree)
        cfl_all.append(conflict)
        conf_all.append(kf.conf[::stride, ::stride].ravel())
        par_all.append(par)
    P = np.vstack(pts_all)
    g = Grade(np.concatenate(cls_all), np.concatenate(agr_all), np.concatenate(cfl_all), np.concatenate(par_all), reasons)
    return P, np.vstack(col_all), g


SUPPORT_RGB = {"R": (37, 99, 235), "I": (245, 158, 11), "U": (120, 120, 120)}


def write_ply(path: str | Path, xyz: F64, rgb: npt.NDArray[np.uint8], extra: dict[str, npt.NDArray[Any]] | None = None) -> None:
    extra = extra or {}
    hdr = ["ply", "format ascii 1.0", f"element vertex {len(xyz)}", "property float x", "property float y", "property float z",
           "property uchar red", "property uchar green", "property uchar blue"] + [f"property uchar {k}" for k in extra] + ["end_header"]
    cols = [np.asarray(v).astype(int) for v in extra.values()]
    lines = [f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}" + "".join(f" {int(e[i])}" for e in cols)
             for i, (p, c) in enumerate(zip(xyz, rgb, strict=True))]
    Path(path).write_text("\n".join(hdr + lines) + "\n", encoding="ascii")


def grade_export(evidence_npz: str | Path, out_dir: str | Path, loss_after_frame: int | None = None, stride: int = 2,
                 lens_status: str = "UNCALIBRATED", scale_status: str = "NO_SCALE") -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kfs = load_evidence(evidence_npz)
    P, rgb, g = grade_keyframes(kfs, loss_after_frame, stride=stride)
    ev_rgb = np.array([SUPPORT_RGB[str(c)] for c in g.classes], dtype=np.uint8)
    write_ply(out / "mast3r_natural.ply", P, rgb)
    write_ply(out / "mast3r_evidence.ply", P, ev_rgb, {"agree": np.minimum(g.agree, 255)})
    counts = {c: int((g.classes == c).sum()) for c in ("R", "I", "U")}
    report = {
        "policy": POLICY, "points": int(len(P)), "keyframes": len(kfs), "stride": stride,
        "keyframe_frame_ids": [kf.frame_id for kf in kfs], "loss_after_frame": loss_after_frame,
        "counts": counts, "fractions": {c: round(counts[c] / max(len(P), 1), 4) for c in counts},
        "reasons": g.reasons, "parallax_deg_p50": round(float(np.median(g.parallax_deg)), 2),
        "reconstructor_conf_p50": round(float(np.median(np.concatenate([kf.conf.ravel() for kf in kfs]))), 2),
        "statuses": {"geometry_support": "per-point R/I/U in mast3r_evidence.ply", "lens_calibration": lens_status,
                     "metric_scale": scale_status},
        "files": {"natural": str(out / "mast3r_natural.ply"), "evidence": str(out / "mast3r_evidence.ply")},
        "notes": ["LAB R means multi-keyframe self-consistent geometry; it is NOT production R/M and no real "
                  "input is promoted to production R or M by this module",
                  "real-sweep class fractions are unvalidated output until a phantom with known geometry grades them",
                  "depth agreement and calibration-free intrinsics come from the reconstructor itself; only colour "
                  "agreement and parallax are independent of it",
                  "reconstructor confidence reported, never used to promote",
                  "R requires >= 2 other keyframes agreeing in depth and colour with >= 2 deg parallax",
                  "depth conflict (another keyframe sees a nearer surface) -> U: drifted sheet or occluder",
                  "measurement record, not a diagnosis"],
    }
    import json

    (out / "grading_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
