# UNFILED-IP
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DemoConfig:
    sensor_height: int = 48
    sensor_width: int = 64
    scale: int = 4
    frame_count: int = 16
    root_seed: int = 20260803
    psf_sigma_lr_truth: float = 0.82
    psf_sigma_lr_estimated: float = 0.70
    read_noise_std: float = 0.004
    shot_electrons: float = 700.0
    saturation_level: float = 1.0
    regularization: float = 0.0025
    max_iterations: int = 36
    gate_z2_threshold: float = 8.0
    tile_size_hr: int = 16

    @property
    def hr_shape(self) -> tuple[int, int]:
        return (self.sensor_height * self.scale, self.sensor_width * self.scale)
