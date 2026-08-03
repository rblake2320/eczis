# UNFILED-IP
from __future__ import annotations

import math

import numpy as np

from eczis.schemas import BoolArray, FloatArray


def rmse(reference: FloatArray, estimate: FloatArray) -> float:
    return float(np.sqrt(np.mean((reference.astype(np.float64) - estimate.astype(np.float64)) ** 2)))


def psnr(reference: FloatArray, estimate: FloatArray) -> float:
    error = rmse(reference, estimate)
    return 99.0 if error == 0.0 else float(20.0 * math.log10(1.0 / error))


def feature_scores(
    reference_mask: BoolArray,
    estimate: FloatArray,
    *,
    threshold: float = 0.36,
) -> dict[str, float]:
    predicted = estimate >= np.float32(threshold)
    tp = int((predicted & reference_mask).sum())
    fp = int((predicted & ~reference_mask).sum())
    fn = int((~predicted & reference_mask).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "feature_precision": precision,
        "feature_recall": recall,
        "feature_f1": f1,
        "false_feature_rate": fp / max(1, int((~reference_mask).sum())),
    }


def challenge_detection_rate(injection_mask: BoolArray, base_pass: BoolArray, challenged_pass: BoolArray) -> float:
    eligible = injection_mask & base_pass
    if not eligible.any():
        return 0.0
    return float((eligible & ~challenged_pass).sum() / eligible.sum())
