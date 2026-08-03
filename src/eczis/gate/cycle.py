# UNFILED-IP
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter, zoom

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.gate.residuals import noise_normalized_residual
from eczis.gate.thresholds import GateThresholds
from eczis.schemas import CaptureSet, FloatArray, GateResult


def consistency_gate(
    proposal_hr: FloatArray,
    capture: CaptureSet,
    model: EstimatedCameraModel,
    thresholds: GateThresholds,
) -> GateResult:
    score_lr, score_hr = noise_normalized_residual(proposal_hr, capture, model)
    # Region aggregation prevents a single noisy pixel from deciding a class.
    smoothed = uniform_filter(score_hr, size=max(3, capture.calibration.scale * 2 + 1), mode="nearest")
    valid_count_lr = np.zeros(capture.calibration.sensor_shape, dtype=np.int16)
    for frame in capture.frames:
        valid_count_lr += frame.valid_mask.astype(np.int16)
    valid_hr = zoom(
        valid_count_lr,
        zoom=capture.calibration.scale,
        order=0,
        mode="nearest",
        prefilter=False,
    )
    enough = valid_hr >= thresholds.min_valid_frames
    passed = (smoothed <= np.float32(thresholds.z2_threshold)) & enough
    reasons = {
        "residual_fail_pixels": int((smoothed > thresholds.z2_threshold).sum()),
        "insufficient_evidence_pixels": int((~enough).sum()),
        "pass_pixels": int(passed.sum()),
    }
    return GateResult(score_lr, smoothed.astype(np.float32), passed.astype(np.bool_), reasons)
