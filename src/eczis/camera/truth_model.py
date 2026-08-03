# UNFILED-IP
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, shift

from eczis.camera.operators import block_average, illumination_map
from eczis.schemas import FloatArray, ObservationParams, RawFrame


class TruthCameraModel:
    """Richer hidden simulator used only to synthesize observations.

    It deliberately differs from EstimatedCameraModel in boundary handling,
    spatially varying blur, mild nonlinearity, and noise generation.
    """

    def __init__(
        self,
        *,
        scale: int,
        psf_sigma_lr: float,
        read_noise_std: float,
        shot_electrons: float,
        saturation_level: float,
    ) -> None:
        self.scale = scale
        self.psf_sigma_lr = psf_sigma_lr
        self.read_noise_std = read_noise_std
        self.shot_electrons = shot_electrons
        self.saturation_level = saturation_level

    def project(self, scene_hr: FloatArray, observation: ObservationParams) -> RawFrame:
        rng = np.random.default_rng(observation.seed)
        illum = illumination_map(
            scene_hr.shape,
            observation.illumination_x * 1.08,
            observation.illumination_y * 0.92,
        )
        weighted = scene_hr * illum * np.float32(observation.exposure_scale)
        shifted = shift(
            weighted,
            shift=(observation.dy_lr * self.scale, observation.dx_lr * self.scale),
            order=3,
            mode="reflect",
            prefilter=True,
        ).astype(np.float32)

        sigma_base = (self.psf_sigma_lr + observation.focus_sigma_delta_lr) * self.scale
        left = gaussian_filter(shifted, sigma=max(0.05, sigma_base * 0.93), mode="reflect")
        right = gaussian_filter(shifted, sigma=max(0.05, sigma_base * 1.11), mode="reflect")
        blend = np.linspace(0.0, 1.0, scene_hr.shape[1], dtype=np.float32)[None, :]
        blurred = left * (1.0 - blend) + right * blend

        sensor = block_average(blurred.astype(np.float32), self.scale)
        sensor = sensor + np.float32(0.012) * sensor * sensor
        sensor = np.clip(sensor, 0.0, self.saturation_level)

        expected_e = sensor * np.float32(self.shot_electrons)
        shot = rng.poisson(expected_e).astype(np.float32) / np.float32(self.shot_electrons)
        read = rng.normal(0.0, self.read_noise_std, sensor.shape).astype(np.float32)
        noisy = np.clip(shot + read, 0.0, self.saturation_level).astype(np.float32)
        saturation = noisy >= np.float32(self.saturation_level * 0.999)
        valid = np.isfinite(noisy) & ~saturation
        return RawFrame(noisy, valid.astype(np.bool_), saturation.astype(np.bool_), observation)
