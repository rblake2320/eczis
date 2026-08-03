# UNFILED-IP
from __future__ import annotations

import numpy as np


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = max(1, confidence.size)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if not selected.any():
            continue
        ece += float(selected.sum()) / total * abs(
            float(confidence[selected].mean()) - float(correct[selected].mean())
        )
    return ece
