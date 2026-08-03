# UNFILED-IP
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GateThresholds:
    z2_threshold: float
    min_valid_frames: int = 4
