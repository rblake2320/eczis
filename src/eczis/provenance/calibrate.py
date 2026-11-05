# UNFILED-IP
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correctness must share a shape")
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


@dataclass(frozen=True, slots=True)
class HistogramCalibrator:
    """Monotone histogram-binning calibrator fitted on the calibration split only."""

    edges: tuple[float, ...]
    values: tuple[float, ...]

    def apply(self, confidence: np.ndarray) -> np.ndarray:
        index = np.clip(np.searchsorted(np.asarray(self.edges[1:-1]), confidence, side="right"), 0, len(self.values) - 1)
        return np.asarray(self.values, dtype=np.float32)[index]


def fit_histogram_calibrator(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> HistogramCalibrator:
    if confidence.size == 0:
        raise ValueError("cannot fit a calibrator on empty data")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.searchsorted(edges[1:-1], confidence.ravel(), side="right"), 0, bins - 1)
    flat_correct = correct.ravel().astype(np.float64)
    values: list[float] = []
    overall = float(flat_correct.mean())
    for b in range(bins):
        selected = index == b
        values.append(float(flat_correct[selected].mean()) if selected.any() else overall)
    # Pool-adjacent-violators: enforce monotone non-decreasing calibrated confidence.
    blocks = [[v, 1.0] for v in values]
    merged: list[list[float]] = []
    for value, weight in blocks:
        merged.append([value, weight])
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            v2, w2 = merged.pop()
            v1, w1 = merged.pop()
            merged.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    monotone: list[float] = []
    for value, weight in merged:
        monotone.extend([value] * int(weight))
    return HistogramCalibrator(tuple(float(e) for e in edges), tuple(monotone))
