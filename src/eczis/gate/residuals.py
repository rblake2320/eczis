# UNFILED-IP
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter, zoom

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.camera.noise import predicted_variance
from eczis.schemas import CaptureSet, FloatArray


@dataclass(frozen=True, slots=True)
class ResidualMaps:
    """Noise-normalized squared residuals.

    ``mean_lr`` aggregates all valid frames; ``max_frame_lr`` is the worst
    single frame after local averaging, so one contradicting observation
    cannot be diluted by the agreeing majority (K-1 attack).
    """

    mean_lr: FloatArray
    max_frame_lr: FloatArray
    worst_frame_index_lr: np.ndarray


def _upsample(score_lr: FloatArray, scale: int, *, order: int) -> FloatArray:
    return np.asarray(zoom(score_lr, zoom=scale, order=order, mode="nearest", prefilter=False), dtype=np.float32)


def residual_maps(
    proposal_hr: FloatArray,
    capture: CaptureSet,
    model: EstimatedCameraModel,
    *,
    frame_window_lr: int = 3,
) -> ResidualMaps:
    shape = capture.calibration.sensor_shape
    accum = np.zeros(shape, dtype=np.float32)
    count = np.zeros_like(accum)
    worst = np.zeros(shape, dtype=np.float32)
    worst_index = np.full(shape, -1, dtype=np.int32)
    for index, frame in enumerate(capture.frames):
        predicted = model.project(proposal_hr, frame.params)
        variance = predicted_variance(
            predicted,
            shot_electrons=capture.calibration.shot_electrons,
            read_noise_std=capture.calibration.read_noise_std,
        )
        squared_z = (((frame.data - predicted) ** 2) / variance).astype(np.float32)
        valid = frame.valid_mask.astype(np.float32)
        accum += squared_z * valid
        count += valid
        # Local mean over valid samples only; invalid samples never count as evidence.
        num = uniform_filter(squared_z * valid, size=frame_window_lr, mode="nearest")
        den = uniform_filter(valid, size=frame_window_lr, mode="nearest")
        local = np.where(den > 0, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
        update = local > worst
        worst = np.where(update, local, worst)
        worst_index = np.where(update, index, worst_index)
    mean_lr = (accum / np.maximum(count, 1.0)).astype(np.float32)
    return ResidualMaps(mean_lr, worst.astype(np.float32), worst_index)


def noise_normalized_residual(
    proposal_hr: FloatArray,
    capture: CaptureSet,
    model: EstimatedCameraModel,
) -> tuple[FloatArray, FloatArray]:
    """Return mean normalized squared residual at LR and HR resolution."""
    maps = residual_maps(proposal_hr, capture, model)
    return maps.mean_lr, _upsample(maps.mean_lr, capture.calibration.scale, order=1)
