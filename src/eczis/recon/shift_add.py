# UNFILED-IP
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from eczis.camera.operators import fourier_shift, illumination_map
from eczis.schemas import CaptureSet, FloatArray


def bicubic_baseline(capture: CaptureSet) -> FloatArray:
    scale = capture.calibration.scale
    image = zoom(capture.frames[0].data, zoom=scale, order=3, mode="reflect", prefilter=True)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def shift_add_reconstruct(capture: CaptureSet) -> FloatArray:
    scale = capture.calibration.scale
    h, w = capture.calibration.sensor_shape
    hr_shape = (h * scale, w * scale)
    accum = np.zeros(hr_shape, dtype=np.float32)
    weights = np.zeros(hr_shape, dtype=np.float32)
    for frame in capture.frames:
        lifted = zoom(frame.data, zoom=scale, order=1, mode="reflect", prefilter=False).astype(np.float32)
        illum = illumination_map(hr_shape, frame.params.illumination_x, frame.params.illumination_y)
        lifted = lifted / np.maximum(illum * np.float32(frame.params.exposure_scale), 1e-4)
        aligned = fourier_shift(
            lifted,
            -frame.params.dx_lr * scale,
            -frame.params.dy_lr * scale,
        )
        valid = zoom(frame.valid_mask.astype(np.float32), zoom=scale, order=0, prefilter=False)
        valid = fourier_shift(valid.astype(np.float32), -frame.params.dx_lr * scale, -frame.params.dy_lr * scale)
        valid = np.clip(valid, 0.0, 1.0)
        accum += aligned * valid
        weights += valid
    result = accum / np.maximum(weights, 1e-5)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
