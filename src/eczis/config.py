# UNFILED-IP
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


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
    gate_max_frame_z2_threshold: float | None = 24.0
    tile_size_hr: int = 16

    def __post_init__(self) -> None:
        if self.scale < 1 or self.sensor_height < 4 or self.sensor_width < 4:
            raise ValueError("invalid sensor geometry")
        side = int(round(self.frame_count**0.5))
        if side * side != self.frame_count or self.frame_count < 4:
            raise ValueError("frame_count must be a square >= 4")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.shot_electrons <= 0 or self.read_noise_std < 0:
            raise ValueError("invalid noise parameters")

    @property
    def hr_shape(self) -> tuple[int, int]:
        return (self.sensor_height * self.scale, self.sensor_width * self.scale)


def load_config(path: Path | None, **overrides: Any) -> DemoConfig:
    """Load a DemoConfig from JSON; unknown keys fail closed."""
    data: dict[str, Any] = {}
    if path is not None:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("config file must contain a JSON object")
        data.update(loaded)
    data.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in fields(DemoConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    return DemoConfig(**data)
