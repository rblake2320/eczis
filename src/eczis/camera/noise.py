# UNFILED-IP
"""Poisson-Gaussian sensor noise: one definition shared by simulator and gate."""

from __future__ import annotations

import numpy as np

from eczis.schemas import FloatArray

VARIANCE_FLOOR = 1e-7


def predicted_variance(signal: FloatArray, *, shot_electrons: float, read_noise_std: float) -> FloatArray:
    """Per-sample variance of a normalized Poisson-Gaussian sensor reading."""
    if shot_electrons <= 0:
        raise ValueError("shot_electrons must be positive")
    if read_noise_std < 0:
        raise ValueError("read_noise_std must be non-negative")
    variance = (
        np.maximum(signal, 0.0) / np.float32(shot_electrons)
        + np.float32(read_noise_std**2)
        + np.float32(VARIANCE_FLOOR)
    )
    return variance.astype(np.float32)


def sample_poisson_gaussian(
    signal: FloatArray,
    rng: np.random.Generator,
    *,
    shot_electrons: float,
    read_noise_std: float,
) -> FloatArray:
    """Draw a noisy normalized reading: Poisson(signal*e)/e + N(0, read^2)."""
    if shot_electrons <= 0:
        raise ValueError("shot_electrons must be positive")
    expected_e = np.maximum(signal, 0.0) * np.float32(shot_electrons)
    shot = rng.poisson(expected_e).astype(np.float32) / np.float32(shot_electrons)
    read = rng.normal(0.0, read_noise_std, signal.shape).astype(np.float32)
    return (shot + read).astype(np.float32)
