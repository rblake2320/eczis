# UNFILED-IP
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter, zoom

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.gate.residuals import _upsample, residual_maps
from eczis.gate.thresholds import GateThresholds
from eczis.schemas import CaptureSet, FloatArray, GateResult


def consistency_gate(
    proposal_hr: FloatArray,
    capture: CaptureSet,
    model: EstimatedCameraModel,
    thresholds: GateThresholds,
) -> GateResult:
    if proposal_hr.ndim != 2 or not np.all(np.isfinite(proposal_hr)):
        raise ValueError("Gate proposal must be a finite 2D array")
    scale = capture.calibration.scale
    h, w = capture.calibration.sensor_shape
    if proposal_hr.shape != (h * scale, w * scale):
        raise ValueError("Gate proposal shape does not match calibration sensor_shape * scale")
    maps = residual_maps(proposal_hr, capture, model)
    score_hr = _upsample(maps.mean_lr, scale, order=1)
    # Region aggregation prevents a single noisy pixel from deciding a class.
    smoothed = uniform_filter(score_hr, size=max(3, scale * 2 + 1), mode="nearest")
    valid_count_lr = np.zeros(capture.calibration.sensor_shape, dtype=np.int16)
    for frame in capture.frames:
        valid_count_lr += frame.valid_mask.astype(np.int16)
    valid_hr = zoom(valid_count_lr, zoom=scale, order=0, mode="nearest", prefilter=False)
    enough = valid_hr >= thresholds.min_valid_frames
    mean_fail = smoothed > np.float32(thresholds.z2_threshold)
    if thresholds.max_frame_z2_threshold is None:
        frame_fail_lr = np.zeros(capture.calibration.sensor_shape, dtype=np.bool_)
    else:
        frame_fail_lr = maps.max_frame_lr > np.float32(thresholds.max_frame_z2_threshold)
    frame_fail = zoom(frame_fail_lr.astype(np.uint8), zoom=scale, order=0, prefilter=False).astype(np.bool_)
    passed = ~mean_fail & ~frame_fail & enough
    flagged_frames = sorted({int(i) for i in np.unique(maps.worst_frame_index_lr[frame_fail_lr]) if i >= 0})
    reasons = {
        "residual_fail_pixels": int(mean_fail.sum()),
        "single_frame_contradiction_pixels": int((frame_fail & ~mean_fail).sum()),
        "insufficient_evidence_pixels": int((~enough).sum()),
        "pass_pixels": int(passed.sum()),
        "contradicting_frame_count": len(flagged_frames),
    }
    return GateResult(maps.mean_lr, smoothed.astype(np.float32), passed.astype(np.bool_), reasons)
