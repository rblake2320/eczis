# UNFILED-IP
from __future__ import annotations

from eczis.config import DemoConfig
from eczis.schemas import Calibration


def calibration_from_config(config: DemoConfig) -> Calibration:
    return Calibration(
        schema_version="1.0",
        scale=config.scale,
        sensor_shape=(config.sensor_height, config.sensor_width),
        estimated_psf_sigma_lr=config.psf_sigma_lr_estimated,
        read_noise_std=config.read_noise_std,
        shot_electrons=config.shot_electrons,
        saturation_level=config.saturation_level,
    )
