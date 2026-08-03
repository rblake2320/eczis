# UNFILED-IP
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.schemas import CaptureSet, FloatArray


def noise_normalized_residual(
    proposal_hr: FloatArray,
    capture: CaptureSet,
    model: EstimatedCameraModel,
) -> tuple[FloatArray, FloatArray]:
    """Return mean normalized squared residual at LR and HR resolution."""
    accum = np.zeros(capture.calibration.sensor_shape, dtype=np.float32)
    count = np.zeros_like(accum)
    for frame in capture.frames:
        predicted = model.project(proposal_hr, frame.params)
        variance = (
            np.maximum(predicted, 0.0) / np.float32(capture.calibration.shot_electrons)
            + np.float32(capture.calibration.read_noise_std**2)
            + np.float32(1e-7)
        )
        squared_z = ((frame.data - predicted) ** 2) / variance
        valid = frame.valid_mask.astype(np.float32)
        accum += squared_z.astype(np.float32) * valid
        count += valid
    score_lr = accum / np.maximum(count, 1.0)
    score_hr = zoom(
        score_lr,
        zoom=capture.calibration.scale,
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32)
    return score_lr.astype(np.float32), score_hr
