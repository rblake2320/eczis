# UNFILED-IP
"""Optional Fourier-feature coordinate-field renderer (roadmap R3).

Each tile is fitted independently by ridge regression on deterministic
Fourier features (seeded). Provenance is never derived from the field: a
query returns the conservative minimum of the source class at that point,
the scale-support class, and a fit-quality class (tiles whose fit error
exceeds ``max_fit_rmse`` are capped at I). The deterministic tiled raster
remains the release baseline; this backend can only demote.
"""

from __future__ import annotations

import math

import numpy as np

from eczis.schemas import FloatArray, ProvenanceClass, QueryResult, UInt8Array


class CoordinateFieldRenderer:
    def __init__(
        self,
        image: FloatArray,
        provenance: UInt8Array,
        confidence: FloatArray,
        scale: int,
        *,
        tile: int = 16,
        features: int = 48,
        bandwidth: float = 0.12,
        ridge: float = 1e-5,
        seed: int = 0,
        max_fit_rmse: float = 0.02,
    ) -> None:
        if image.shape != provenance.shape or image.shape != confidence.shape:
            raise ValueError("Renderer arrays must share a shape")
        if tile < 4 or features < 4:
            raise ValueError("tile and features must be >= 4")
        self.image = image.astype(np.float32)
        self.provenance = provenance
        self.confidence = confidence
        self.native_scale = scale
        self.tile = tile
        self.ridge = ridge
        self.max_fit_rmse = max_fit_rmse
        rng = np.random.default_rng(seed)
        self._omega = rng.normal(0.0, bandwidth * 2.0 * math.pi, size=(features, 2))
        self._phase = rng.uniform(0.0, 2.0 * math.pi, size=features)
        self._cache: dict[tuple[int, int], tuple[np.ndarray, float, int, int]] = {}

    def _design(self, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        proj = np.outer(yy, self._omega[:, 0]) + np.outer(xx, self._omega[:, 1]) + self._phase
        scale = float(self.tile)
        return np.hstack(
            [np.cos(proj), np.sin(proj), np.ones((yy.size, 1)), (yy / scale)[:, None], (xx / scale)[:, None]]
        )

    def _fit(self, ty: int, tx: int) -> tuple[np.ndarray, float, int, int]:
        key = (ty, tx)
        if key in self._cache:
            return self._cache[key]
        h, w = self.image.shape
        y0, x0 = ty * self.tile, tx * self.tile
        y1, x1 = min(h, y0 + self.tile), min(w, x0 + self.tile)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        ly, lx = (yy - y0).ravel().astype(np.float64), (xx - x0).ravel().astype(np.float64)
        design = self._design(ly, lx)
        target = self.image[y0:y1, x0:x1].ravel().astype(np.float64)
        gram = design.T @ design + self.ridge * np.eye(design.shape[1])
        weights = np.linalg.solve(gram, design.T @ target)
        rmse = float(np.sqrt(np.mean((design @ weights - target) ** 2)))
        self._cache[key] = (weights, rmse, y0, x0)
        return self._cache[key]

    def fit_rmse(self, x: float, y: float) -> float:
        h, w = self.image.shape
        py = float(np.clip(y, 0.0, 1.0) * (h - 1))
        px = float(np.clip(x, 0.0, 1.0) * (w - 1))
        return self._fit(int(py) // self.tile, int(px) // self.tile)[1]

    def query(self, x: float, y: float, scale: float) -> QueryResult:
        h, w = self.image.shape
        px = float(np.clip(x, 0.0, 1.0) * (w - 1))
        py = float(np.clip(y, 0.0, 1.0) * (h - 1))
        weights, rmse, y0, x0 = self._fit(int(py) // self.tile, int(px) // self.tile)
        value = float((self._design(np.array([py - y0]), np.array([px - x0])) @ weights)[0])
        ix, iy = int(round(px)), int(round(py))
        base_class = ProvenanceClass(int(self.provenance[iy, ix]))
        confidence = float(self.confidence[iy, ix])
        if scale > self.native_scale * 3.0:
            scale_class = ProvenanceClass.U
            confidence *= math.exp(-(scale / self.native_scale - 1.5))
        elif scale > self.native_scale * 1.5:
            scale_class = ProvenanceClass.I
            confidence *= math.exp(-(scale / self.native_scale - 1.5))
        else:
            scale_class = ProvenanceClass.M
        fit_class = ProvenanceClass.M if rmse <= self.max_fit_rmse else ProvenanceClass.I
        final = ProvenanceClass(min(int(base_class), int(scale_class), int(fit_class)))
        return QueryResult(value, final, float(np.clip(confidence, 0.0, 1.0)), x, y, scale)
