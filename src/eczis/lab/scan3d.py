# UNFILED-IP
"""EXPERIMENTAL evidence-aware 3D scan of a canal (ear canal, nostril, pipe) from one moving camera.

Pipeline: track corners through the video (Lucas-Kanade) -> initialise two-view geometry
(essential matrix) -> register later frames by PnP -> triangulate -> grade every 3D point by how it
was observed (views, triangulation angle, reprojection error) into R / I / U -> canal coverage map
and next-motion advice. Monocular scale is unknowable from images alone; it comes from one known
baseline (in practice an insertion-depth mark or a printed marker of known size).

A synthetic canal renderer provides ground truth for validation. Measurement record only: not a
diagnosis. Synthetic software reduction-to-practice until validated on physical targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


# --------------------------------------------------------------------------- synthetic canal
@dataclass(frozen=True, slots=True)
class Canal:
    """Tube along +z with radius r(theta, z) in mm (defaults ~ an adult ear canal)."""

    r0: float = 3.5
    length: float = 25.0
    narrow_z: float = 14.0  # z of the narrowing ("isthmus")
    narrow_frac: float = 0.25
    bump_theta: float = 1.2  # a raised lump on the wall (e.g. wax / swelling stand-in)
    bump_z: float = 9.0
    bump_mm: float = 0.6
    seed: int = 11

    def radius(self, theta: F64, z: F64) -> F64:
        r = self.r0 * (1.0 - self.narrow_frac * np.exp(-(((z - self.narrow_z) / 3.0) ** 2)))
        dth = np.angle(np.exp(1j * (theta - self.bump_theta)))
        r = r - self.bump_mm * np.exp(-(dth / 0.35) ** 2 - ((z - self.bump_z) / 1.2) ** 2)
        return np.asarray(r, dtype=np.float64)

    def texture(self) -> F64:
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(self.seed)
        tex = gaussian_filter(rng.random((512, 512)), 1.2, mode="wrap")
        tex = (tex - tex.min()) / (tex.max() - tex.min())
        return np.asarray(0.35 + 0.65 * tex, dtype=np.float64)  # (theta, z) albedo map


@dataclass(frozen=True, slots=True)
class Intrinsics:
    fx: float = 160.0
    fy: float = 160.0
    cx: float = 160.0
    cy: float = 120.0
    width: int = 320
    height: int = 240

    def matrix(self) -> F64:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


def look_rotation(yaw: float, pitch: float, roll: float) -> F64:
    """World-from-camera rotation: camera +z looks down the canal, perturbed by yaw/pitch, twisted by roll."""
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return np.asarray(ry @ rx @ rz, dtype=np.float64)


def render(canal: Canal, tex: F64, centre: F64, rot_wc: F64, k: Intrinsics) -> F64:
    """Ray-cast one frame: albedo x co-located LED shading (cos / d^2), 8-bit quantised."""
    u, v = np.meshgrid(np.arange(k.width) + 0.5, np.arange(k.height) + 0.5)
    rays_c = np.stack([(u - k.cx) / k.fx, (v - k.cy) / k.fy, np.ones_like(u)], axis=-1)
    rays = rays_c @ rot_wc.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    o = centre[None, None, :]

    def inside(t: F64) -> F64:
        p = o + rays * t[..., None]
        rho = np.hypot(p[..., 0], p[..., 1])
        return rho - canal.radius(np.arctan2(p[..., 1], p[..., 0]), p[..., 2])

    t_lo = np.full(u.shape, 0.05)
    t_hi = np.full(u.shape, np.nan)
    t = t_lo.copy()
    for step in np.arange(0.15, 40.0, 0.15):  # march to the first wall crossing
        hit = np.isnan(t_hi) & (inside(np.full(u.shape, step)) > 0)
        t_hi[hit] = step
        t_lo[np.isnan(t_hi)] = step
        if not np.isnan(t_hi).any():
            break
    miss = np.isnan(t_hi)
    t_hi[miss] = t_lo[miss]
    for _ in range(18):  # bisection
        t = 0.5 * (t_lo + t_hi)
        out = inside(t) > 0
        t_hi = np.where(out, t, t_hi)
        t_lo = np.where(out, t_lo, t)
    p = o + rays * t[..., None]
    theta = np.mod(np.arctan2(p[..., 1], p[..., 0]), 2 * np.pi)
    z = p[..., 2]
    ti = (theta / (2 * np.pi) * tex.shape[0]).astype(int) % tex.shape[0]
    zi = (z / 6.0 * tex.shape[1]).astype(int) % tex.shape[1]  # texture repeats every 6 mm
    albedo = tex[ti, zi]
    radial = -np.stack([p[..., 0], p[..., 1], np.zeros_like(z)], axis=-1)
    radial /= np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), 1e-9)
    cosang = np.clip(np.abs(np.sum(-rays * radial, axis=-1)), 0.05, 1.0)
    img = albedo * cosang * np.clip(9.0 / np.maximum(t, 0.5) ** 2, 0, 1.0)
    img = np.where(miss | (z > canal.length), 0.02, img)
    img = img / max(float(np.percentile(img, 99)), 1e-6) * 0.85  # camera auto-exposure
    return np.asarray(np.round(np.clip(img, 0, 1) * 255) / 255, dtype=np.float64)


@dataclass
class SyntheticScan:
    canal: Canal
    k: Intrinsics
    frames: list[F64]
    centres: list[F64]
    rotations: list[F64]  # world-from-camera


def simulate_sweep(canal: Canal | None = None, frames: int = 60, depth_mm: float = 6.0,
                   twist_deg: float = 45.0, wobble_mm: float = 0.4, k: Intrinsics | None = None) -> SyntheticScan:
    """Insert the camera slowly (z 0..depth) with a twist and small lateral wobble, as by hand."""
    canal = canal or Canal()
    k = k or Intrinsics()
    tex = canal.texture()
    out = SyntheticScan(canal, k, [], [], [])
    for i in range(frames):
        s = i / max(frames - 1, 1)
        c = np.array([wobble_mm * np.sin(2.1 * np.pi * s), wobble_mm * np.cos(1.3 * np.pi * s) - wobble_mm, depth_mm * s])
        r = look_rotation(0.08 * np.sin(np.pi * s), 0.06 * np.cos(np.pi * s), np.radians(twist_deg) * s)
        out.frames.append(render(canal, tex, c, r, k))
        out.centres.append(c)
        out.rotations.append(r)
    return out


# --------------------------------------------------------------------------- reconstruction
@dataclass
class Point3D:
    xyz: F64
    views: int
    angle_deg: float
    reproj_px: float
    support: str = "U"


@dataclass
class ScanResult:
    points: list[Point3D]
    centres: list[F64]  # estimated camera centres (world = first camera)
    rotations: list[F64]
    tracked_frames: int
    notes: list[str] = field(default_factory=list)


def _proj(k: F64, r_cw: F64, t_cw: F64) -> F64:
    return np.asarray(k @ np.hstack([r_cw, t_cw.reshape(3, 1)]), dtype=np.float64)


def reconstruct(frames: list[F64], k: Intrinsics, baseline_mm: float, init_gap: int = 6,
                min_angle_deg: float = 2.0) -> ScanResult:
    """Monocular reconstruction. baseline_mm = true distance between camera 0 and camera init_gap."""
    import cv2

    kmat = k.matrix()
    imgs = [np.ascontiguousarray((np.clip(f, 0, 1) * 255).astype(np.uint8)) for f in frames]
    # --- track corners through the sequence; keep per-track observations
    tracks: dict[int, dict[int, tuple[float, float]]] = {}
    next_id = 0
    pts = cv2.goodFeaturesToTrack(imgs[0], 600, 0.01, 6)
    ids = list(range(len(pts))) if pts is not None else []
    for j, p in zip(ids, pts.reshape(-1, 2) if pts is not None else [], strict=False):
        tracks[j] = {0: (float(p[0]), float(p[1]))}
    next_id = len(ids)
    cur = pts.reshape(-1, 1, 2).astype(np.float32) if pts is not None else np.zeros((0, 1, 2), np.float32)
    for f in range(1, len(imgs)):
        if len(cur):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(imgs[f - 1], imgs[f], cur, None, winSize=(21, 21), maxLevel=3)  # type: ignore[call-overload]
            back, st2, _ = cv2.calcOpticalFlowPyrLK(imgs[f], imgs[f - 1], nxt, None, winSize=(21, 21), maxLevel=3)  # type: ignore[call-overload]
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - cur).reshape(-1, 2), axis=1) < 0.7)
            keep_ids = [i for i, g in zip(ids, good, strict=True) if g]
            cur = nxt[good]
            ids = keep_ids
            for i, p in zip(ids, cur.reshape(-1, 2), strict=True):
                tracks[i][f] = (float(p[0]), float(p[1]))
        if len(ids) < 300:  # replenish features away from existing ones
            mask = np.full(imgs[f].shape, 255, np.uint8)
            for p in cur.reshape(-1, 2):
                cv2.circle(mask, (int(p[0]), int(p[1])), 8, 0, -1)
            new = cv2.goodFeaturesToTrack(imgs[f], 600 - len(ids), 0.01, 6, mask=mask)
            if new is not None:
                for p in new.reshape(-1, 2):
                    tracks[next_id] = {f: (float(p[0]), float(p[1]))}
                    ids.append(next_id)
                    next_id += 1
                cur = np.vstack([cur.reshape(-1, 1, 2), new.reshape(-1, 1, 2)]).astype(np.float32)
    # --- two-view initialisation between frame 0 and init_gap
    common = [i for i, t in tracks.items() if 0 in t and init_gap in t]
    a = np.array([tracks[i][0] for i in common])
    b = np.array([tracks[i][init_gap] for i in common])
    e, inl = cv2.findEssentialMat(a, b, kmat, cv2.RANSAC, 0.999, 1.0)
    _, r1, t1, _ = cv2.recoverPose(e, a, b, kmat, mask=inl)
    t1 = t1.ravel() * baseline_mm  # metric scale from the known baseline
    rcw: dict[int, F64] = {0: np.eye(3), init_gap: np.asarray(r1, dtype=np.float64)}
    tcw: dict[int, F64] = {0: np.zeros(3), init_gap: np.asarray(t1, dtype=np.float64)}
    xyz: dict[int, F64] = {}

    def triangulate(i_ids: list[int], fa: int, fb: int) -> None:
        if not i_ids:
            return
        pa = np.array([tracks[i][fa] for i in i_ids]).T
        pb = np.array([tracks[i][fb] for i in i_ids]).T
        hom = cv2.triangulatePoints(_proj(kmat, rcw[fa], tcw[fa]), _proj(kmat, rcw[fb], tcw[fb]), pa, pb)
        pts3 = (hom[:3] / hom[3]).T
        for i, x in zip(i_ids, pts3, strict=True):
            depth_a = (rcw[fa] @ x + tcw[fa])[2]
            depth_b = (rcw[fb] @ x + tcw[fb])[2]
            if depth_a > 0 and depth_b > 0 and np.all(np.isfinite(x)):
                xyz[i] = x

    triangulate([i for i, ok in zip(common, inl.ravel(), strict=True) if ok], 0, init_gap)
    # --- register remaining frames by PnP, triangulating new tracks against an earlier frame
    order = [f for f in range(1, len(imgs)) if f != init_gap]
    for f in order:
        obs = [i for i in xyz if f in tracks[i]]
        if len(obs) < 12:
            continue
        obj = np.array([xyz[i] for i in obs])
        img = np.array([tracks[i][f] for i in obs])
        ok, rvec, tvec, inl2 = cv2.solvePnPRansac(obj, img, kmat, None, reprojectionError=1.5, iterationsCount=200)
        if not ok or inl2 is None or len(inl2) < 10:
            continue
        rcw[f] = np.asarray(cv2.Rodrigues(rvec)[0], dtype=np.float64)
        tcw[f] = np.asarray(tvec, dtype=np.float64).ravel()
        ref = max((g for g in rcw if g < f), default=None)
        if ref is not None and f - ref >= 1:
            back = max((g for g in rcw if g <= f - 3), default=ref)
            fresh = [i for i, t in tracks.items() if i not in xyz and f in t and back in t]
            triangulate(fresh, back, f)
    # --- grade every point by how it was observed
    points: list[Point3D] = []
    centres = {f: -rcw[f].T @ tcw[f] for f in rcw}
    for i, x in xyz.items():
        seen = [f for f in tracks[i] if f in rcw]
        if len(seen) < 2:
            continue
        errs, dirs = [], []
        for f in seen:
            pc = rcw[f] @ x + tcw[f]
            if pc[2] <= 0:
                continue
            uv = kmat @ (pc / pc[2])
            errs.append(float(np.hypot(*(uv[:2] - np.array(tracks[i][f])))))
            d = x - centres[f]
            dirs.append(d / np.linalg.norm(d))
        if len(dirs) < 2:
            continue
        cosines = np.clip(np.array(dirs) @ np.array(dirs).T, -1, 1)
        angle = float(np.degrees(np.arccos(cosines.min())))
        rms = float(np.sqrt(np.mean(np.square(errs))))
        support = "R" if (len(dirs) >= 3 and angle >= min_angle_deg and rms <= 1.0) else (
            "I" if (angle >= min_angle_deg / 2 and rms <= 2.0) else "U")
        points.append(Point3D(x, len(dirs), angle, rms, support))
    return ScanResult(points, [centres[f] for f in sorted(rcw)], [rcw[f].T for f in sorted(rcw)], len(rcw))


# --------------------------------------------------------------------------- coverage & guidance
def coverage(points: list[Point3D], z_range: tuple[float, float], theta_bins: int = 12, z_bin_mm: float = 1.0) -> dict[str, Any]:
    """Fraction of canal-wall cells (theta x z) holding at least one R point, plus the largest gap."""
    nz = max(1, int(np.ceil((z_range[1] - z_range[0]) / z_bin_mm)))
    grid = np.zeros((theta_bins, nz), dtype=bool)
    for p in points:
        if p.support != "R":
            continue
        th = np.mod(np.arctan2(p.xyz[1], p.xyz[0]), 2 * np.pi)
        zi = int((p.xyz[2] - z_range[0]) // z_bin_mm)
        if 0 <= zi < nz:
            grid[int(th / (2 * np.pi) * theta_bins) % theta_bins, zi] = True
    missing_by_sector = (~grid).sum(axis=1)
    worst = int(np.argmax(missing_by_sector))
    return {"covered_fraction": round(float(grid.mean()), 3), "grid": grid.astype(int).tolist(),
            "least_covered_sector_deg": round((worst + 0.5) * 360 / theta_bins, 1),
            "least_covered_sector_missing_cells": int(missing_by_sector[worst])}


def next_motion(cov: dict[str, Any]) -> str:
    if cov["covered_fraction"] >= 0.9:
        return "Coverage sufficient: stop scanning."
    return (f"Rotate the tip toward {cov['least_covered_sector_deg']:.0f} deg (that wall sector has "
            f"{cov['least_covered_sector_missing_cells']} unseen 1 mm bands), then sweep slowly in and out.")


# --------------------------------------------------------------------------- export
SUPPORT_RGB = {"R": (37, 99, 235), "I": (245, 158, 11), "U": (120, 120, 120)}


def export_ply(points: list[Point3D], path: str) -> int:
    """ASCII PLY point cloud coloured by support class (R blue, I amber, U grey); opens in Blender/MeshLab."""
    from pathlib import Path

    lines = ["ply", "format ascii 1.0", f"element vertex {len(points)}", "property float x", "property float y",
             "property float z", "property uchar red", "property uchar green", "property uchar blue",
             "property uchar views", "end_header"]
    for p in points:
        r, g, b = SUPPORT_RGB[p.support]
        lines.append(f"{p.xyz[0]:.4f} {p.xyz[1]:.4f} {p.xyz[2]:.4f} {r} {g} {b} {min(p.views, 255)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
    return len(points)


def summarize(result: ScanResult, z_range: tuple[float, float] = (2.0, 14.0)) -> dict[str, Any]:
    counts = {c: sum(p.support == c for p in result.points) for c in ("R", "I", "U")}
    cov = coverage(result.points, z_range)
    return {"frames_registered": result.tracked_frames, "points": counts, "coverage": cov["covered_fraction"],
            "least_covered_sector_deg": cov["least_covered_sector_deg"], "advice": next_motion(cov),
            "notes": ["scale from the supplied baseline; measurement record, not a diagnosis"]}
