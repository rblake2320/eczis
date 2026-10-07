# UNFILED-IP
"""Lens calibration from a printed checkerboard (OpenCV pinhole + radial/tangential distortion).

Produces a LensProfile (fx, fy, cx, cy, distortion, reprojection error, views used) which is the
missing link between arbitrary-unit 3D output and real millimetres: with a calibrated focal
length, a known checkerboard square size gives metric scale directly, and the 3D scan can use
fx/fy instead of the "0.8 x width" guess.

A synthetic checkerboard renderer with a known lens lets the calibration be tested without a
print. EXPERIMENTAL; never changes production M/R/I/U.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]

MIN_VIEWS = 8  # 6 solved fx but left cx ~18 px off and k2/k3 unconstrained on synthetic tests
MAX_RMS_PX = 1.0
HELD_OUT_MAX_PX = 1.5  # a view the solver never saw must still reproject within this (or 2 x RMS)
MIN_CENTRE_SPREAD = 0.08  # std of the board centre across views, as a fraction of the image size
MIN_TILT_DEG = 12.0  # at least one view this far from front-facing, or distortion/focal length stay entangled
MIN_SIZE_RATIO = 1.25  # apparent board size must vary (near and far views)


@dataclass(frozen=True, slots=True)
class LensProfile:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...]  # k1 k2 p1 p2 k3
    width: int
    height: int
    rms_px: float
    views: int
    square_mm: float
    source: str = "checkerboard"
    per_view_rms_px: tuple[float, ...] = ()  # one reprojection RMS per accepted view (stability check)
    model: str = "pinhole+k1k2p1p2 (k3 fixed 0)"
    camera: str = ""  # device identity, e.g. "camera 1 USB2.0 PC CAMERA"
    capture_mode: str = ""  # e.g. "640x480 YUY2 MSMF"
    held_out_rms_px: tuple[float, ...] = ()  # leave-one-view-out reprojection RMS: calibrated on the others, scored on it
    diversity: dict[str, float] = field(default_factory=dict)  # board position/tilt/size spread across accepted views
    print_record: str = ""  # printed-target verification this profile's square size came from

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**asdict(self), "dist": list(self.dist), "per_view_rms_px": list(self.per_view_rms_px),
                                 "held_out_rms_px": list(self.held_out_rms_px),
                                 "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")
        return p

    @staticmethod
    def from_json(path: str | Path) -> LensProfile:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d.pop("created", None)
        d["dist"] = tuple(d["dist"])
        d["per_view_rms_px"] = tuple(d.get("per_view_rms_px", ()))
        d["held_out_rms_px"] = tuple(d.get("held_out_rms_px", ()))
        d.setdefault("diversity", {})
        return LensProfile(**d)

    def matrix(self) -> F64:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def ok(self) -> bool:
        return self.views >= MIN_VIEWS and self.rms_px <= MAX_RMS_PX

    def held_out_stable(self) -> bool | None:
        """Held-out error no worse than max(1.5 px, 2 x fitted RMS) on every view; None if not computed."""
        if not self.held_out_rms_px:
            return None
        return max(self.held_out_rms_px) <= max(HELD_OUT_MAX_PX, 2.0 * self.rms_px)

    def diverse(self) -> bool | None:
        d = self.diversity
        if not d:
            return None
        return (d.get("centre_spread_norm", 0.0) >= MIN_CENTRE_SPREAD and d.get("tilt_deg_max", 0.0) >= MIN_TILT_DEG
                and d.get("size_ratio_max_over_min", 1.0) >= MIN_SIZE_RATIO)

    def perturbed(self, frac: float) -> LensProfile:
        """Wrong-calibration control: fx, fy scaled by 1+frac and cx shifted by frac*width/2, labelled as such."""
        return replace(self, fx=self.fx * (1 + frac), fy=self.fy * (1 + frac), cx=self.cx + frac * self.width / 2,
                       source=f"{self.source} PERTURBED {frac:+.0%} (wrong-calibration control)")


@dataclass
class Calibrator:
    """Accumulates checkerboard detections from frames, then solves the lens."""

    cols: int = 9  # inner corners across
    rows: int = 6  # inner corners down
    square_mm: float = 2.0
    detections: list[F64] = field(default_factory=list)
    image_size: tuple[int, int] | None = None

    def _object_points(self) -> F64:
        g = np.zeros((self.rows * self.cols, 3), np.float32)
        g[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        return np.asarray(g * self.square_mm, dtype=np.float32)

    def detect(self, gray: np.ndarray) -> F64 | None:
        """Find the board in one frame; returns refined corners (N,2) or None."""
        import cv2

        img = gray if gray.dtype == np.uint8 else (np.clip(gray, 0, 1) * 255).astype(np.uint8)
        ok, corners = cv2.findChessboardCornersSB(  # type: ignore[call-overload]
            np.ascontiguousarray(img), (self.cols, self.rows), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not ok:
            return None
        self.image_size = (img.shape[1], img.shape[0])
        return np.asarray(corners.reshape(-1, 2), dtype=np.float64)

    def add(self, gray: np.ndarray, min_shift_px: float = 15.0) -> bool:
        """Detect and keep this view if it differs enough from views already kept."""
        c = self.detect(gray)
        if c is None:
            return False
        for prev in self.detections:
            if float(np.mean(np.linalg.norm(prev - c, axis=1))) < min_shift_px:
                return False  # near-duplicate pose adds nothing
        self.detections.append(c)
        return True

    def solve(self, camera: str = "", capture_mode: str = "", held_out_check: bool = True, print_record: str = "") -> LensProfile:
        import cv2

        if len(self.detections) < MIN_VIEWS or self.image_size is None:
            raise ValueError(f"need at least {MIN_VIEWS} distinct board views, have {len(self.detections)}; "
                             "tilt and move the board between captures")
        obj = [self._object_points()] * len(self.detections)
        imgp = [np.asarray(d, dtype=np.float32).reshape(-1, 1, 2) for d in self.detections]
        rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(obj, imgp, self.image_size, None, None, flags=cv2.CALIB_FIX_K3)
        d = np.asarray(dist).ravel()
        per_view = []
        for o, ip, rv, tv in zip(obj, imgp, rvecs, tvecs, strict=True):
            proj, _ = cv2.projectPoints(o, rv, tv, k, dist)
            per_view.append(_rms(proj, ip))
        # held-out: calibrate without view i, pose view i with that lens, score its reprojection
        held_out = []
        if held_out_check:
            for i in range(len(obj)):
                rest_o, rest_i = obj[:i] + obj[i + 1:], imgp[:i] + imgp[i + 1:]
                _, k_i, d_i, _, _ = cv2.calibrateCamera(rest_o, rest_i, self.image_size, None, None, flags=cv2.CALIB_FIX_K3)
                ok, rv, tv = cv2.solvePnP(obj[i], imgp[i], k_i, d_i)
                proj, _ = cv2.projectPoints(obj[i], rv, tv, k_i, d_i)
                held_out.append(_rms(proj, imgp[i]) if ok else float("inf"))
        return LensProfile(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]),
                           tuple(float(v) for v in d[:5]), self.image_size[0], self.image_size[1],
                           float(rms), len(self.detections), self.square_mm, per_view_rms_px=tuple(per_view),
                           camera=camera, capture_mode=capture_mode, held_out_rms_px=tuple(held_out),
                           diversity=self.diversity(rvecs), print_record=print_record)

    def diversity(self, rvecs: Any) -> dict[str, float]:
        """How varied the accepted views are: board centre spread, tilt range, apparent size range, roll range."""
        import cv2

        assert self.image_size is not None
        w, h = self.image_size
        centres = np.array([d.mean(axis=0) / [w, h] for d in self.detections])
        sizes = np.array([float(np.mean(np.linalg.norm(np.diff(d[: self.cols], axis=0), axis=1))) for d in self.detections])
        tilts, rolls = [], []
        for rv in rvecs:
            r, _ = cv2.Rodrigues(np.asarray(rv, dtype=np.float64))
            tilts.append(float(np.degrees(np.arccos(np.clip(abs(r[2, 2]), -1, 1)))))
            rolls.append(float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))))
        return {"centre_spread_norm": round(float(np.mean(centres.std(axis=0))), 4),
                "size_ratio_max_over_min": round(float(sizes.max() / max(sizes.min(), 1e-9)), 3),
                "tilt_deg_min": round(min(tilts), 1), "tilt_deg_max": round(max(tilts), 1),
                "roll_deg_range": round(max(rolls) - min(rolls), 1)}


# --------------------------------------------------------------------------- synthetic board
def render_board(k: F64, dist: F64, rvec: F64, tvec: F64, size: tuple[int, int], cols: int = 9, rows: int = 6,
                 square_mm: float = 2.0, noise: float = 0.01) -> np.ndarray:
    """Render a checkerboard through a known lens (used to validate the calibrator)."""
    import cv2

    w, h = size
    # project a fine grid of board points and rasterise squares by nearest sample
    n = 8  # samples per square
    gx = np.linspace(-1, cols, (cols + 2) * n)
    gy = np.linspace(-1, rows, (rows + 2) * n)
    bx, by = np.meshgrid(gx, gy)
    pts = np.stack([bx.ravel() * square_mm, by.ravel() * square_mm, np.zeros(bx.size)], axis=1).astype(np.float64)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, k, dist)
    proj = proj.reshape(-1, 2)
    shade = (((np.floor(bx) + np.floor(by)) % 2) == 0).ravel()  # chequer colour per sample
    img = np.full((h, w), 0.55, np.float64)  # grey background
    # splat: for each projected sample paint a small disc; order dark then light to keep edges crisp
    yy, xx = np.mgrid[0:h, 0:w]
    rad = max(1.0, 0.45 * np.median(np.linalg.norm(np.diff(proj.reshape((rows + 2) * n, (cols + 2) * n, 2), axis=1), axis=2)))
    for light in (False, True):
        sel = proj[shade == light]
        sel = sel[(sel[:, 0] > -rad) & (sel[:, 0] < w + rad) & (sel[:, 1] > -rad) & (sel[:, 1] < h + rad)]
        for px, py in sel:
            x0, x1 = int(max(0, px - rad)), int(min(w, px + rad + 1))
            y0, y1 = int(max(0, py - rad)), int(min(h, py + rad + 1))
            if x1 > x0 and y1 > y0:
                m = (xx[y0:y1, x0:x1] - px) ** 2 + (yy[y0:y1, x0:x1] - py) ** 2 <= rad * rad
                img[y0:y1, x0:x1][m] = 0.9 if light else 0.1
    img = np.asarray(cv2.GaussianBlur(img, (0, 0), 0.7), dtype=np.float64)
    rng = np.random.default_rng(int(abs(tvec[2]) * 1000) % 2**31)
    return np.asarray(np.clip(img + rng.normal(0, noise, img.shape), 0, 1), dtype=np.float64)


def synthetic_views(k: F64, dist: F64, size: tuple[int, int], count: int = 10, seed: int = 0) -> list[np.ndarray]:
    """Board seen from varied tilts/distances through lens k, dist."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        rvec = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), rng.uniform(-0.3, 0.3)])
        tvec = np.array([rng.uniform(-6, 2), rng.uniform(-4, 1), rng.uniform(28, 45)])
        out.append(render_board(k, dist, rvec, tvec, size))
    return out


def _rms(proj: Any, ip: Any) -> float:
    return float(np.sqrt(np.mean(np.sum((np.asarray(proj).reshape(-1, 2) - np.asarray(ip).reshape(-1, 2)) ** 2, axis=1))))


def summary(profile: LensProfile) -> dict[str, Any]:
    return {"fx": round(profile.fx, 2), "fy": round(profile.fy, 2), "cx": round(profile.cx, 2), "cy": round(profile.cy, 2),
            "k1": round(profile.dist[0], 4), "k2": round(profile.dist[1], 4), "rms_px": round(profile.rms_px, 3),
            "views": profile.views, "ok": profile.ok(),
            "per_view_rms_px": [round(v, 3) for v in profile.per_view_rms_px],
            "worst_view_rms_px": round(max(profile.per_view_rms_px), 3) if profile.per_view_rms_px else None,
            "held_out_rms_px": [round(v, 3) for v in profile.held_out_rms_px],
            "held_out_max_px": round(max(profile.held_out_rms_px), 3) if profile.held_out_rms_px else None,
            "held_out_stable": profile.held_out_stable(), "diversity": profile.diversity, "diverse": profile.diverse(),
            "hfov_deg": round(float(2 * np.degrees(np.arctan(profile.width / (2 * profile.fx)))), 1)}
