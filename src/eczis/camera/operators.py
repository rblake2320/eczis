# UNFILED-IP
from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def fourier_shift(image: FloatArray, dx_px: float, dy_px: float) -> FloatArray:
    """Periodic subpixel shift using the Fourier shift theorem."""
    h, w = image.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    phase = np.exp(-2j * np.pi * (fx * dx_px + fy * dy_px))
    shifted = np.fft.ifft2(np.fft.fft2(image) * phase).real
    return shifted.astype(np.float32)


def gaussian_blur_periodic(image: FloatArray, sigma_px: float) -> FloatArray:
    if sigma_px <= 0:
        return image.copy()
    h, w = image.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    transfer = np.exp(-2.0 * (np.pi**2) * sigma_px**2 * (fx**2 + fy**2))
    blurred = np.fft.ifft2(np.fft.fft2(image) * transfer).real
    return blurred.astype(np.float32)


def block_average(image: FloatArray, scale: int) -> FloatArray:
    h, w = image.shape
    if h % scale or w % scale:
        raise ValueError("High-resolution shape must be divisible by scale")
    return image.reshape(h // scale, scale, w // scale, scale).mean(axis=(1, 3)).astype(np.float32)


def block_average_adjoint(image_lr: FloatArray, scale: int) -> FloatArray:
    return (np.repeat(np.repeat(image_lr, scale, axis=0), scale, axis=1) / float(scale * scale)).astype(
        np.float32
    )


def illumination_map(shape: tuple[int, int], x_slope: float, y_slope: float) -> FloatArray:
    h, w = shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    illum = 1.0 + x_slope * xx + y_slope * yy
    return np.clip(illum, 0.65, 1.35).astype(np.float32)
