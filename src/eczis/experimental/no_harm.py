# UNFILED-IP
"""EXPERIMENTAL: no-harm reconstruction selector (truth-free).

A reconstruction may replace a declared baseline only if it predicts held-out frames better.
For each held-out frame k, every candidate is rebuilt WITHOUT frame k, projected into frame k's
geometry, and scored by the noise-normalised squared residual. The candidate is selected only if
its mean held-out score beats the baseline's by the margin; otherwise the baseline is returned and
the rejection is recorded. Not used by production classification.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.schemas import CaptureSet, FloatArray

Builder = Callable[[CaptureSet], FloatArray]


@dataclass(frozen=True, slots=True)
class NoHarmDecision:
    selected: str
    scores: dict[str, float]
    baseline: str
    reason: str


def held_out_score(build: Builder, capture: CaptureSet, model: EstimatedCameraModel, holdout: Sequence[int]) -> float:
    scores = []
    for k in holdout:
        kept = [f for i, f in enumerate(capture.frames) if i != k]
        estimate = build(replace(capture, frames=kept))
        frame = capture.frames[k]
        pred = model.project(estimate.astype(np.float32), frame.params)
        var = np.maximum(pred, 0.0) / capture.calibration.shot_electrons + capture.calibration.read_noise_std**2 + 1e-7
        valid = frame.valid_mask
        scores.append(float(np.mean(((frame.data - pred) ** 2 / var)[valid])))
    return float(np.mean(scores))


def no_harm_select(
    candidates: dict[str, Builder],
    baseline: str,
    capture: CaptureSet,
    model: EstimatedCameraModel,
    holdout: Sequence[int],
    margin: float = 0.05,
) -> NoHarmDecision:
    scores = {name: held_out_score(b, capture, model, holdout) for name, b in candidates.items()}
    best = min(scores, key=lambda n: scores[n])
    if best != baseline and scores[best] < scores[baseline] * (1.0 - margin):
        return NoHarmDecision(best, scores, baseline, f"{best} beats {baseline} on held-out frames")
    return NoHarmDecision(baseline, scores, baseline, "no candidate beat the baseline by the margin")
