# UNFILED-IP
from __future__ import annotations

from dataclasses import dataclass

from eczis.schemas import BoolArray, FloatArray


@dataclass(frozen=True, slots=True)
class Target:
    target_id: str
    image: FloatArray
    feature_mask: BoolArray
    feature_count: int
