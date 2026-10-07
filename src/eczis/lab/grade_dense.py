# UNFILED-IP
"""Evidence-grade a dense point cloud from an external reconstructor (e.g. MASt3R-SLAM).

The external tool supplies points and a camera trajectory but no per-point evidence. ECZIS grades
every point by its own observability: how many keyframe cameras had it in view (in front of the
camera and inside the image when projected with the lens), and the angular spread of those
viewing directions. Nothing here promotes a point; it only assigns R / I / U so that dense,
confident-looking surfaces carry the same honesty as the sparse tracker's output.

Rules (research policy constants): R needs >= 3 views and >= 2 deg of viewing-angle spread;
I needs >= 2 views; everything else is U. Metric scale is only as good as the input's.
EXPERIMENTAL; never changes production M/R/I/U.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]

R_MIN_VIEWS, R_MIN_ANGLE_DEG = 3, 2.0
I_MIN_VIEWS = 2


@dataclass(frozen=True, slots=True)
class Pose:
    t: float
    position: F64  # camera centre in world
    rot_wc: F64  # world-from-camera rotation (3x3)


def read_tum_trajectory(path: str | Path) -> list[Pose]:
    """TUM format lines: t x y z qx qy qz qw (world-from-camera)."""
    poses = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 8 or parts[0].startswith("#"):
            continue
        t, x, y, z, qx, qy, qz, qw = (float(v) for v in parts)
        n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
        rot = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        poses.append(Pose(t, np.array([x, y, z]), rot))
    return poses


def read_ply(path: str | Path) -> tuple[F64, npt.NDArray[np.uint8] | None]:
    """Binary little-endian or ASCII PLY with x y z [red green blue] float/uchar vertex props."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    end += 2 if raw[end : end + 2] == b"\r\n" else 1
    header = raw[:end].decode("ascii", errors="replace")
    n = int(next(ln for ln in header.splitlines() if ln.startswith("element vertex")).split()[-1])
    props = [ln.split()[1:] for ln in header.splitlines() if ln.startswith("property ")]
    names = [p[1] for p in props]
    fmt = {"float": "f", "float32": "f", "double": "d", "uchar": "B", "uint8": "B", "int": "i", "uint": "I"}
    ascii_mode = "format ascii" in header
    if ascii_mode:
        rows = np.loadtxt(raw[end:].decode("ascii").splitlines()[:n], dtype=np.float64, ndmin=2)
    else:
        st = "<" + "".join(fmt[p[0]] for p in props)
        rows = np.array(list(struct.iter_unpack(st, raw[end : end + n * struct.calcsize(st)])), dtype=np.float64)
    xyz = rows[:, [names.index("x"), names.index("y"), names.index("z")]]
    rgb = rows[:, [names.index(c) for c in ("red", "green", "blue")]].astype(np.uint8) if "red" in names else None
    return np.asarray(xyz, dtype=np.float64), rgb


def grade_points(xyz: F64, poses: list[Pose], k: F64, image_size: tuple[int, int], chunk: int = 20000) -> dict[str, Any]:
    """Count, per point, the keyframes that see it and the angular spread; assign R/I/U."""
    w, h = image_size
    n = xyz.shape[0]
    views = np.zeros(n, dtype=np.int32)
    dir_sum = np.zeros((n, 3), dtype=np.float64)
    min_cos = np.ones(n, dtype=np.float64)
    first_dir = np.zeros((n, 3), dtype=np.float64)
    seen_any = np.zeros(n, dtype=bool)
    for pose in poses:
        rot_cw = pose.rot_wc.T
        for s in range(0, n, chunk):
            p = xyz[s : s + chunk]
            cam = (p - pose.position) @ rot_cw.T  # world -> camera
            z = cam[:, 2]
            ok = z > 1e-6
            u = k[0, 0] * cam[:, 0] / np.where(ok, z, 1.0) + k[0, 2]
            v = k[1, 1] * cam[:, 1] / np.where(ok, z, 1.0) + k[1, 2]
            ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
            d = p - pose.position
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            idx = np.flatnonzero(ok) + s
            views[idx] += 1
            dir_sum[idx] += d[ok]
            new = idx[~seen_any[idx]]
            first_dir[new] = d[ok][~seen_any[idx]]
            seen_any[new] = True
            cos = np.einsum("ij,ij->i", d[ok], first_dir[idx])
            min_cos[idx] = np.minimum(min_cos[idx], cos)
    spread_deg = np.degrees(np.arccos(np.clip(min_cos, -1.0, 1.0)))
    cls = np.full(n, "U", dtype="<U1")
    cls[views >= I_MIN_VIEWS] = "I"
    cls[(views >= R_MIN_VIEWS) & (spread_deg >= R_MIN_ANGLE_DEG)] = "R"
    counts = {c: int((cls == c).sum()) for c in ("R", "I", "U")}
    return {"classes": cls, "views": views, "spread_deg": spread_deg, "counts": counts,
            "fractions": {c: round(counts[c] / max(n, 1), 4) for c in counts}, "keyframes": len(poses), "points": n}


SUPPORT_RGB = {"R": (37, 99, 235), "I": (245, 158, 11), "U": (120, 120, 120)}


def write_graded_ply(path: str | Path, xyz: F64, classes: npt.NDArray[Any], views: npt.NDArray[np.int32]) -> int:
    lines = ["ply", "format ascii 1.0", f"element vertex {len(xyz)}", "property float x", "property float y",
             "property float z", "property uchar red", "property uchar green", "property uchar blue", "property uchar views",
             "end_header"]
    for p, c, v in zip(xyz, classes, views, strict=True):
        r, g, b = SUPPORT_RGB[str(c)]
        lines.append(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {r} {g} {b} {min(int(v), 255)}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
    return len(xyz)


def grade_mast3r_output(ply: str | Path, traj: str | Path, k: F64, image_size: tuple[int, int], out_ply: str | Path | None = None) -> dict[str, Any]:
    xyz, _ = read_ply(ply)
    poses = read_tum_trajectory(traj)
    if not poses:
        raise ValueError(f"no poses in {traj}")
    g = grade_points(xyz, poses, k, image_size)
    if out_ply is not None:
        write_graded_ply(out_ply, xyz, g["classes"], g["views"])
    return {"points": g["points"], "keyframes": g["keyframes"], "counts": g["counts"], "fractions": g["fractions"],
            "graded_ply": None if out_ply is None else str(out_ply),
            "views_min_max": [int(g["views"].min()), int(g["views"].max())] if g["points"] else [0, 0],
            "spread_deg_p50": round(float(np.median(g["spread_deg"])), 1) if g["points"] else 0.0,
            "notes": ["grading by keyframe visibility and viewing-angle spread only; no per-point reprojection "
                      "check yet (the reconstructor does not save per-keyframe depth/confidence)",
                      "the reconstructor already dropped points below its own confidence threshold before saving",
                      "if every point is seen by every keyframe, visibility cannot discriminate and R means "
                      "'seen from several angles', not 'verified consistent'",
                      "scale and lens as supplied to the reconstructor"]}
