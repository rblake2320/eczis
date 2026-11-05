# UNFILED-IP
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """Registered gate thresholds.

    ``z2_threshold`` bounds the region mean of the noise-normalized squared
    residual over all frames. ``max_frame_z2_threshold`` bounds the worst
    single frame (3x3 LR local mean); ``None`` disables that criterion and is
    only for ablation studies. Default 24 was registered from clean
    smooth-target runs whose per-frame p99.9 was <= 5 at three SNR regimes
    (see docs/VALIDATION_PROTOCOL.md).
    """

    z2_threshold: float
    min_valid_frames: int = 4
    max_frame_z2_threshold: float | None = 24.0

    def __post_init__(self) -> None:
        if not self.z2_threshold > 0:
            raise ValueError("z2_threshold must be positive")
        if self.min_valid_frames < 1:
            raise ValueError("min_valid_frames must be >= 1")
        if self.max_frame_z2_threshold is not None and not self.max_frame_z2_threshold > 0:
            raise ValueError("max_frame_z2_threshold must be positive or None")
