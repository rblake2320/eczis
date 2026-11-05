# UNFILED-IP
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.provenance.observability import (
    DEFAULT_MAX_UNSUPPORTED_RMS,
    DEFAULT_MIN_TRANSFER,
    identifiable_mask,
    observability_map,
)
from eczis.schemas import CaptureSet, FloatArray, GateResult, ProvenanceClass, UInt8Array

R_MIN_OBSERVABILITY = 0.42
R_MAX_PRIOR = 0.05


def classify_provenance(
    capture: CaptureSet,
    gate: GateResult,
    *,
    proposal_hr: FloatArray,
    model: EstimatedCameraModel,
    prior_contribution: FloatArray | None = None,
    min_transfer: float = DEFAULT_MIN_TRANSFER,
    max_unsupported_rms: float = DEFAULT_MAX_UNSUPPORTED_RMS,
) -> tuple[UInt8Array, FloatArray]:
    """Conservative M/R/I/U classification.

    R requires all of: gate pass, sampling observability, prior below the R
    limit, valid samples, and *identifiability* -- the proposal's local
    out-of-band energy is below ``max_unsupported_rms``. Detail the forward
    operator cannot see stays I however low its residual (stop condition 6).
    M is never emitted: reconstructed HR pixels are not direct samples.
    """
    obs = observability_map(capture)
    if proposal_hr.shape != obs.shape:
        raise ValueError("Proposal shape must match the HR evidence grid")
    prior = np.zeros_like(obs) if prior_contribution is None else prior_contribution
    if prior.shape != obs.shape:
        raise ValueError("Prior contribution shape must match the HR evidence grid")

    invalid_lr = np.ones(capture.calibration.sensor_shape, dtype=np.bool_)
    for frame in capture.frames:
        invalid_lr &= ~frame.valid_mask
    invalid_hr = zoom(
        invalid_lr.astype(np.uint8),
        zoom=capture.calibration.scale,
        order=0,
        prefilter=False,
    ).astype(np.bool_)
    identifiable = identifiable_mask(
        proposal_hr, model, min_transfer=min_transfer, max_unsupported_rms=max_unsupported_rms
    )

    classes = np.full(obs.shape, int(ProvenanceClass.I), dtype=np.uint8)
    reconstructable = (
        gate.pass_mask_hr
        & (obs >= R_MIN_OBSERVABILITY)
        & (prior <= R_MAX_PRIOR)
        & identifiable
        & ~invalid_hr
    )
    classes[reconstructable] = int(ProvenanceClass.R)
    classes[invalid_hr] = int(ProvenanceClass.U)

    confidence = np.exp(-gate.score_hr / np.float32(12.0)) * obs * (1.0 - np.clip(prior, 0, 1))
    confidence = np.where(identifiable, confidence, confidence * np.float32(0.5))
    confidence[invalid_hr] = 0.0
    return classes, np.clip(confidence, 0.0, 1.0).astype(np.float32)


def demote(base: UInt8Array, limiter: UInt8Array) -> UInt8Array:
    """Monotonic provenance composition: lower class always wins."""
    if base.shape != limiter.shape:
        raise ValueError("Provenance maps must have matching shapes")
    return np.minimum(base, limiter).astype(np.uint8)
