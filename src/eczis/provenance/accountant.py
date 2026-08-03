# UNFILED-IP
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from eczis.provenance.observability import observability_map
from eczis.schemas import CaptureSet, FloatArray, GateResult, ProvenanceClass, UInt8Array


def classify_provenance(
    capture: CaptureSet,
    gate: GateResult,
    *,
    prior_contribution: FloatArray | None = None,
) -> tuple[UInt8Array, FloatArray]:
    obs = observability_map(capture)
    prior = np.zeros_like(obs) if prior_contribution is None else prior_contribution

    invalid_lr = np.ones(capture.calibration.sensor_shape, dtype=np.bool_)
    for frame in capture.frames:
        invalid_lr &= ~frame.valid_mask
    invalid_hr = zoom(
        invalid_lr.astype(np.uint8),
        zoom=capture.calibration.scale,
        order=0,
        prefilter=False,
    ).astype(np.bool_)

    classes = np.full(obs.shape, int(ProvenanceClass.I), dtype=np.uint8)
    classes[invalid_hr] = int(ProvenanceClass.U)
    reconstructable = gate.pass_mask_hr & (obs >= 0.42) & (prior <= 0.05) & ~invalid_hr
    classes[reconstructable] = int(ProvenanceClass.R)
    classes[(~gate.pass_mask_hr) & ~invalid_hr] = int(ProvenanceClass.I)

    confidence = np.exp(-gate.score_hr / np.float32(12.0)) * obs * (1.0 - np.clip(prior, 0, 1))
    confidence[invalid_hr] = 0.0
    return classes, np.clip(confidence, 0.0, 1.0).astype(np.float32)


def demote(base: UInt8Array, limiter: UInt8Array) -> UInt8Array:
    """Monotonic provenance composition: lower class always wins."""
    if base.shape != limiter.shape:
        raise ValueError("Provenance maps must have matching shapes")
    return np.minimum(base, limiter).astype(np.uint8)
