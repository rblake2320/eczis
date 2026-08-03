# UNFILED-IP
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from eczis.schemas import CaptureSet, FloatArray


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
