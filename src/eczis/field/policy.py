# UNFILED-IP
"""Single source of truth for render-time provenance rules.

The Python renderer and the exported offline viewer both consume this policy, so a
coordinate inspected in either place receives the same effective class.

Thresholds are TEMPORARY CONSERVATIVE POLICY CONSTANTS, not measured local
observability. They are expressed as magnification relative to the reconstruction
grid (1.0 = one reconstruction pixel per display pixel).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from eczis.schemas import ProvenanceClass


@dataclass(frozen=True, slots=True)
class ScaleDemotionPolicy:
    demote_to_inferred_above: float = 1.5
    demote_to_unsupported_above: float = 3.0
    confidence_decay_start: float = 1.5
    basis: str = "temporary conservative policy constants; not measured local observability"

    def scale_cap(self, magnification: float) -> ProvenanceClass:
        if magnification > self.demote_to_unsupported_above:
            return ProvenanceClass.U
        if magnification > self.demote_to_inferred_above:
            return ProvenanceClass.I
        return ProvenanceClass.M

    def confidence_factor(self, magnification: float) -> float:
        if magnification > self.confidence_decay_start:
            return math.exp(-(magnification - self.confidence_decay_start))
        return 1.0

    def to_payload(self, native_scale: int) -> dict[str, Any]:
        return {
            "order": [c.name for c in sorted(ProvenanceClass)],
            "native_scale": native_scale,
            "magnification_units": "display pixels per reconstruction pixel",
            "demote_to_inferred_above": self.demote_to_inferred_above,
            "demote_to_unsupported_above": self.demote_to_unsupported_above,
            "confidence_decay_start": self.confidence_decay_start,
            "footprint": "bilinear; weakest class over every contributor with nonzero weight",
            "basis": self.basis,
        }


DEFAULT_POLICY = ScaleDemotionPolicy()


def bilinear_footprint(
    px: float, py: float, height: int, width: int
) -> list[tuple[int, int, float]]:
    """Contributors (row, col, weight) of a bilinear sample at pixel-centre coordinates.

    Coordinates are clamped to the image (edge replication); contributors whose weight
    is exactly zero are excluded, so an integer query uses only its own pixel.
    """
    px = min(max(px, 0.0), float(width - 1))
    py = min(max(py, 0.0), float(height - 1))
    x0, y0 = int(math.floor(px)), int(math.floor(py))
    fx, fy = px - x0, py - y0
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    candidates = (
        (y0, x0, (1.0 - fx) * (1.0 - fy)),
        (y0, x1, fx * (1.0 - fy)),
        (y1, x0, (1.0 - fx) * fy),
        (y1, x1, fx * fy),
    )
    merged: dict[tuple[int, int], float] = {}
    for row, col, weight in candidates:
        if weight > 0.0:
            merged[(row, col)] = merged.get((row, col), 0.0) + weight
    return [(row, col, weight) for (row, col), weight in merged.items()]
