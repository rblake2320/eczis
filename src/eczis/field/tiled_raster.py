# UNFILED-IP
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import map_coordinates

from eczis.schemas import FloatArray, ProvenanceClass, QueryResult, UInt8Array


class TiledRasterRenderer:
    """Deterministic continuous renderer that can only preserve or demote evidence."""

    def __init__(self, image: FloatArray, provenance: UInt8Array, confidence: FloatArray, scale: int) -> None:
        if image.shape != provenance.shape or image.shape != confidence.shape:
            raise ValueError("Renderer arrays must share a shape")
        self.image = image
        self.provenance = provenance
        self.confidence = confidence
        self.native_scale = scale

    def query(self, x: float, y: float, scale: float) -> QueryResult:
        h, w = self.image.shape
        px = float(np.clip(x, 0.0, 1.0) * (w - 1))
        py = float(np.clip(y, 0.0, 1.0) * (h - 1))
        value = float(map_coordinates(self.image, [[py], [px]], order=1, mode="nearest")[0])
        ix, iy = int(round(px)), int(round(py))
        base_class = ProvenanceClass(int(self.provenance[iy, ix]))
        confidence = float(self.confidence[iy, ix])
        # Querying beyond supported reconstruction bandwidth cannot remain R/M.
        if scale > self.native_scale * 3.0:
            scale_class = ProvenanceClass.U
            confidence *= math.exp(-(scale / self.native_scale - 1.5))
        elif scale > self.native_scale * 1.5:
            scale_class = ProvenanceClass.I
            confidence *= math.exp(-(scale / self.native_scale - 1.5))
        else:
            scale_class = base_class
        final_class = ProvenanceClass(min(int(base_class), int(scale_class)))
        return QueryResult(value, final_class, float(np.clip(confidence, 0, 1)), x, y, scale)
