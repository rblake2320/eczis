# UNFILED-IP
from __future__ import annotations

import numpy as np

from eczis.field.policy import DEFAULT_POLICY, ScaleDemotionPolicy, bilinear_footprint
from eczis.schemas import FloatArray, ProvenanceClass, QueryResult, UInt8Array


class TiledRasterRenderer:
    """Deterministic continuous renderer that can only preserve or demote evidence.

    Each rendered sample carries the weakest provenance class among every reconstruction
    pixel with nonzero interpolation weight, further capped by the shared scale policy
    (``eczis.field.policy``), which the exported viewer also consumes.
    """

    def __init__(
        self,
        image: FloatArray,
        provenance: UInt8Array,
        confidence: FloatArray,
        scale: int,
        policy: ScaleDemotionPolicy = DEFAULT_POLICY,
    ) -> None:
        if image.shape != provenance.shape or image.shape != confidence.shape:
            raise ValueError("Renderer arrays must share a shape")
        self.image = image
        self.provenance = provenance
        self.confidence = confidence
        self.native_scale = scale
        self.policy = policy

    def query(self, x: float, y: float, scale: float) -> QueryResult:
        """Query at normalized coordinates in [0, 1]; ``scale`` is relative to the sensor grid."""
        h, w = self.image.shape
        px = float(np.clip(x, 0.0, 1.0) * (w - 1))
        py = float(np.clip(y, 0.0, 1.0) * (h - 1))
        result = self.query_pixel(px, py, scale)
        return QueryResult(result.value, result.provenance, result.confidence, x, y, scale)

    def query_pixel(self, px: float, py: float, scale: float) -> QueryResult:
        """Query at pixel-centre reconstruction coordinates; ``scale`` is relative to the sensor grid."""
        h, w = self.image.shape
        footprint = bilinear_footprint(px, py, h, w)
        value = sum(float(self.image[r, c]) * weight for r, c, weight in footprint)
        base_class = min(int(self.provenance[r, c]) for r, c, _ in footprint)
        confidence = min(float(self.confidence[r, c]) for r, c, _ in footprint)
        magnification = scale / float(self.native_scale)
        final_class = min(base_class, int(self.policy.scale_cap(magnification)))
        confidence *= self.policy.confidence_factor(magnification)
        return QueryResult(
            float(value), ProvenanceClass(final_class), float(np.clip(confidence, 0, 1)), px, py, scale
        )
