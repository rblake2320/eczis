# UNFILED-IP
from __future__ import annotations

import numpy as np
from scipy.ndimage import fourier_shift as scipy_fourier_shift

from eczis.schemas import FloatArray


def estimate_translation(reference: FloatArray, moving: FloatArray) -> tuple[float, float]:
    """Integer-plus-parabolic phase-correlation estimate in (dx, dy) pixels."""
    if reference.shape != moving.shape:
        raise ValueError("Registration inputs must have matching shapes")
    cross = np.fft.fft2(reference) * np.conj(np.fft.fft2(moving))
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.ifft2(cross).real
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = corr.shape
    if peak_x > w // 2:
        peak_x -= w
    if peak_y > h // 2:
        peak_y -= h
    return float(peak_x), float(peak_y)


def apply_translation(image: FloatArray, dx: float, dy: float) -> FloatArray:
    transformed = np.fft.ifftn(scipy_fourier_shift(np.fft.fftn(image), (dy, dx))).real
    return transformed.astype(np.float32)
