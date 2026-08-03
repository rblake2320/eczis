# UNFILED-IP
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
UInt8Array = npt.NDArray[np.uint8]


class ProvenanceClass(IntEnum):
    U = 0
    I = 1  # noqa: E741 — canonical M/R/I/U provenance class name (Inferred)
    R = 2
    M = 3


@dataclass(frozen=True, slots=True)
class ObservationParams:
    frame_id: str
    dx_lr: float
    dy_lr: float
    focus_sigma_delta_lr: float
    exposure_scale: float
    illumination_x: float
    illumination_y: float
    seed: int


@dataclass(slots=True)
class RawFrame:
    data: FloatArray
    valid_mask: BoolArray
    saturation_mask: BoolArray
    params: ObservationParams

    def validate(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("RAW frame must be a 2D array")
        if self.data.shape != self.valid_mask.shape or self.data.shape != self.saturation_mask.shape:
            raise ValueError("RAW frame masks must match data shape")
        if self.data.dtype != np.float32:
            raise TypeError("RAW data must be float32")


@dataclass(frozen=True, slots=True)
class Calibration:
    schema_version: str
    scale: int
    sensor_shape: tuple[int, int]
    estimated_psf_sigma_lr: float
    read_noise_std: float
    shot_electrons: float
    saturation_level: float


@dataclass(slots=True)
class CaptureSet:
    schema_version: str
    target_id: str
    root_seed: int
    calibration: Calibration
    frames: list[RawFrame]
    truth_digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.frames:
            raise ValueError("Capture set must contain frames")
        expected = self.calibration.sensor_shape
        for frame in self.frames:
            frame.validate()
            if frame.data.shape != expected:
                raise ValueError("Frame shape differs from calibration sensor shape")


@dataclass(slots=True)
class GateResult:
    score_lr: FloatArray
    score_hr: FloatArray
    pass_mask_hr: BoolArray
    reasons: dict[str, int]


@dataclass(slots=True)
class ReconArtifacts:
    reconstruction: FloatArray
    bicubic: FloatArray
    shift_add: FloatArray
    residual_hr: FloatArray
    provenance: UInt8Array
    confidence: FloatArray
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryResult:
    value: float
    provenance: ProvenanceClass
    confidence: float
    x: float
    y: float
    scale: float


@dataclass(frozen=True, slots=True)
class VerifyReport:
    valid: bool
    errors: tuple[str, ...]
    manifest_digest: str | None
