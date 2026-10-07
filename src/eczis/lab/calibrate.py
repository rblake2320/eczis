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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]

MIN_VIEWS = 8  # 6 solved fx but left cx ~18 px off and k2/k3 unconstrained on synthetic tests
MAX_RMS_PX = 1.0


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

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**asdict(self), "dist": list(self.dist), "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                indent=2), encoding="utf-8")
        return p

    @staticmethod
    def from_json(path: str | Path) -> LensProfile:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d.pop("created", None)
        d["dist"] = tuple(d["dist"])
        return LensProfile(**d)

    def matrix(self) -> F64:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def ok(self) -> bool:
        return self.views >= MIN_VIEWS and self.rms_px <= MAX_RMS_PX


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

    def solve(self) -> LensProfile:
        import cv2

        if len(self.detections) < MIN_VIEWS or self.image_size is None:
            raise ValueError(f"need at least {MIN_VIEWS} distinct board views, have {len(self.detections)}; "
                             "tilt and move the board between captures")
        obj = [self._object_points()] * len(self.detections)
        imgp = [np.asarray(d, dtype=np.float32).reshape(-1, 1, 2) for d in self.detections]
        rms, k, dist, _, _ = cv2.calibrateCamera(obj, imgp, self.image_size, None, None, flags=cv2.CALIB_FIX_K3)
        d = np.asarray(dist).ravel()
        return LensProfile(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]),
                           tuple(float(v) for v in d[:5]), self.image_size[0], self.image_size[1],
                           float(rms), len(self.detections), self.square_mm)


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


def summary(profile: LensProfile) -> dict[str, Any]:
    return {"fx": round(profile.fx, 2), "fy": round(profile.fy, 2), "cx": round(profile.cx, 2), "cy": round(profile.cy, 2),
            "k1": round(profile.dist[0], 4), "k2": round(profile.dist[1], 4), "rms_px": round(profile.rms_px, 3),
            "views": profile.views, "ok": profile.ok(),
            "hfov_deg": round(float(2 * np.degrees(np.arctan(profile.width / (2 * profile.fx)))), 1)}
