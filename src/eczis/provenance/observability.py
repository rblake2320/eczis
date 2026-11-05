# UNFILED-IP
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.schemas import BoolArray, CaptureSet, FloatArray

# Registered identifiability parameters (docs/VALIDATION_PROTOCOL.md).
DEFAULT_MIN_TRANSFER = 0.10
DEFAULT_MAX_UNSUPPORTED_RMS = 0.02
DEFAULT_ENERGY_WINDOW_SIGMA_HR = 4.0


def observability_map(capture: CaptureSet) -> FloatArray:
    """Conservative approximation using valid support and subpixel phase diversity."""
    scale = capture.calibration.scale
    phases = np.array([(f.params.dx_lr % 1.0, f.params.dy_lr % 1.0) for f in capture.frames])
    phase_x = min(1.0, float(np.std(phases[:, 0]) * 4.0))
    phase_y = min(1.0, float(np.std(phases[:, 1]) * 4.0))
    diversity = np.float32(np.sqrt(phase_x * phase_y))
    support_lr = np.zeros(capture.calibration.sensor_shape, dtype=np.float32)
    for frame in capture.frames:
        support_lr += frame.valid_mask.astype(np.float32)
    support_lr /= float(len(capture.frames))
    support_hr = zoom(support_lr, zoom=scale, order=1, mode="nearest", prefilter=False)
    return np.asarray(np.clip(support_hr * diversity, 0.0, 1.0), dtype=np.float32)


def transfer_magnitude(shape_hr: tuple[int, int], model: EstimatedCameraModel) -> FloatArray:
    """|H(f)| of the estimated model: Gaussian PSF times the pixel box (no defocus credit)."""
    h, w = shape_hr
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    sigma_hr = model.psf_sigma_lr * model.scale
    gaussian = np.exp(-2.0 * (math.pi**2) * sigma_hr**2 * (fx**2 + fy**2))
    box = np.abs(np.sinc(model.scale * fx) * np.sinc(model.scale * fy))
    return np.asarray(gaussian * box, dtype=np.float32)


def supported_band(
    shape_hr: tuple[int, int], model: EstimatedCameraModel, *, min_transfer: float = DEFAULT_MIN_TRANSFER
) -> BoolArray:
    """Frequencies the estimated forward operator transmits above ``min_transfer``.

    Content outside this band is (near) null-space: observations barely
    constrain it, so a low cycle residual says nothing about its truth.
    """
    if not 0.0 < min_transfer < 1.0:
        raise ValueError("min_transfer must be in (0, 1)")
    return np.asarray(transfer_magnitude(shape_hr, model) >= np.float32(min_transfer), dtype=np.bool_)


def unsupported_detail_rms(
    proposal_hr: FloatArray,
    model: EstimatedCameraModel,
    *,
    min_transfer: float = DEFAULT_MIN_TRANSFER,
    window_sigma_hr: float = DEFAULT_ENERGY_WINDOW_SIGMA_HR,
) -> FloatArray:
    """Local RMS of the proposal's out-of-band (non-identifiable) component."""
    band = supported_band(proposal_hr.shape, model, min_transfer=min_transfer)
    spectrum = np.fft.fft2(proposal_hr.astype(np.float64))
    in_band = np.fft.ifft2(spectrum * band).real
    out_of_band = proposal_hr.astype(np.float64) - in_band
    energy = gaussian_filter(out_of_band**2, sigma=window_sigma_hr, mode="nearest")
    return np.asarray(np.sqrt(np.maximum(energy, 0.0)), dtype=np.float32)


def identifiable_mask(
    proposal_hr: FloatArray,
    model: EstimatedCameraModel,
    *,
    min_transfer: float = DEFAULT_MIN_TRANSFER,
    max_unsupported_rms: float = DEFAULT_MAX_UNSUPPORTED_RMS,
) -> BoolArray:
    rms = unsupported_detail_rms(proposal_hr, model, min_transfer=min_transfer)
    return np.asarray(rms <= np.float32(max_unsupported_rms), dtype=np.bool_)
