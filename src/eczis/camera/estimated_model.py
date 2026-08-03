# UNFILED-IP
from __future__ import annotations

import numpy as np

from eczis.camera.operators import (
    block_average,
    block_average_adjoint,
    fourier_shift,
    gaussian_blur_periodic,
    illumination_map,
)
from eczis.schemas import FloatArray, ObservationParams


class EstimatedCameraModel:
    """Linear, calibrated reconstruction model intentionally simpler than truth."""

    def __init__(self, *, scale: int, psf_sigma_lr: float) -> None:
        self.scale = scale
        self.psf_sigma_lr = psf_sigma_lr

    def project(self, scene_hr: FloatArray, observation: ObservationParams) -> FloatArray:
        illum = illumination_map(scene_hr.shape, observation.illumination_x, observation.illumination_y)
        weighted = scene_hr * illum * np.float32(observation.exposure_scale)
        shifted = fourier_shift(
            weighted,
            observation.dx_lr * self.scale,
            observation.dy_lr * self.scale,
        )
        sigma_hr = (self.psf_sigma_lr + observation.focus_sigma_delta_lr) * self.scale
        blurred = gaussian_blur_periodic(shifted, max(0.05, sigma_hr))
        return block_average(blurred, self.scale)

    def adjoint(self, residual_lr: FloatArray, observation: ObservationParams) -> FloatArray:
        lifted = block_average_adjoint(residual_lr, self.scale)
        sigma_hr = (self.psf_sigma_lr + observation.focus_sigma_delta_lr) * self.scale
        blurred = gaussian_blur_periodic(lifted, max(0.05, sigma_hr))
        unshifted = fourier_shift(
            blurred,
            -observation.dx_lr * self.scale,
            -observation.dy_lr * self.scale,
        )
        illum = illumination_map(unshifted.shape, observation.illumination_x, observation.illumination_y)
        return (unshifted * illum * np.float32(observation.exposure_scale)).astype(np.float32)
