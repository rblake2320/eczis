# UNFILED-IP
"""One synthetic case end to end: target -> capture -> recon -> gate -> provenance."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from eczis.bench.metrics import challenge_detection_rate, feature_scores, psnr, rmse
from eczis.camera.calibration import calibration_from_config
from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.camera.truth_model import TruthCameraModel
from eczis.config import DemoConfig
from eczis.gate.cycle import consistency_gate
from eczis.gate.thresholds import GateThresholds
from eczis.provenance.accountant import classify_provenance
from eczis.provenance.calibrate import expected_calibration_error
from eczis.provenance.observability import DEFAULT_MIN_TRANSFER, supported_band, transfer_magnitude
from eczis.recon.map_solver import SolverReport, map_reconstruct
from eczis.recon.shift_add import bicubic_baseline, shift_add_reconstruct
from eczis.rig.schedule import lattice_schedule
from eczis.rig.simulated import SimulatedRig
from eczis.schemas import CaptureSet, FloatArray, GateResult, ProvenanceClass, UInt8Array
from eczis.targets import (
    create_coin_target,
    create_iris_target,
    create_microglyph_target,
    create_pcb_target,
    create_siemens_target,
)
from eczis.targets.base import Target

# Ground-truth definition of a *false* R pixel: local RMS error against the
# hidden truth *restricted to the supported band* above this tolerance
# (Gaussian window sigma 2 HR px). R asserts the observable (in-band)
# component only; out-of-band truth is unobservable by construction and is
# reported separately as ``false_r_rate_full_band`` -- never hidden.
R_ERROR_TOLERANCE = 0.08
R_ERROR_WINDOW_SIGMA = 2.0


def make_target(family: str, shape: tuple[int, int], seed: int) -> Target:
    if family == "microglyph":
        return create_microglyph_target(shape, seed=seed, stroke_px=1)
    if family == "siemens":
        return create_siemens_target(shape)
    if family == "coin":
        return create_coin_target(shape)
    if family == "iris":
        return create_iris_target(shape, seed=seed)
    if family == "pcb":
        return create_pcb_target(shape)
    raise ValueError(f"unknown target family {family!r}")


def truth_model_for(cfg: DemoConfig) -> TruthCameraModel:
    return TruthCameraModel(
        scale=cfg.scale,
        psf_sigma_lr=cfg.psf_sigma_lr_truth,
        read_noise_std=cfg.read_noise_std,
        shot_electrons=cfg.shot_electrons,
        saturation_level=cfg.saturation_level,
    )


def estimated_model_for(cfg: DemoConfig) -> EstimatedCameraModel:
    return EstimatedCameraModel(scale=cfg.scale, psf_sigma_lr=cfg.psf_sigma_lr_estimated)


def thresholds_for(cfg: DemoConfig) -> GateThresholds:
    return GateThresholds(
        cfg.gate_z2_threshold,
        min_valid_frames=max(4, cfg.frame_count // 4),
        max_frame_z2_threshold=cfg.gate_max_frame_z2_threshold,
    )


def simulate_capture(target: Target, cfg: DemoConfig) -> CaptureSet:
    truth = truth_model_for(cfg)
    estimated = estimated_model_for(cfg)
    truth_cls: type[object] = truth.__class__
    estimated_cls: type[object] = estimated.__class__
    if truth_cls is estimated_cls:
        raise RuntimeError("Stop condition: truth and estimated models are identical")
    schedule = lattice_schedule(cfg.frame_count, cfg.root_seed)
    return SimulatedRig(truth, target, calibration_from_config(cfg)).capture_burst(
        schedule, root_seed=cfg.root_seed
    )


def single_frame_deconvolution(
    capture: CaptureSet, model: EstimatedCameraModel, cfg: DemoConfig
) -> tuple[FloatArray, SolverReport]:
    """Phase 4 baseline: the same regularized solver restricted to frame 0."""
    single = replace(capture, frames=capture.frames[:1], metadata={**capture.metadata, "baseline": "single_frame"})
    return map_reconstruct(single, model, regularization=cfg.regularization, max_iterations=cfg.max_iterations)


def inject_false_feature(image: FloatArray) -> tuple[FloatArray, np.ndarray, float]:
    """Inject a false microglyph-like H whose projected energy is detectable."""
    challenged = image.copy()
    h, w = image.shape
    sy, sx = max(1, h // 192), max(1, w // 256)
    y0, x0 = h // 2 - 10 * sy, w // 2 - 14 * sx
    mask = np.zeros_like(image, dtype=np.bool_)
    mask[y0 : y0 + 22 * sy, x0 : x0 + 3 * sx] = True
    mask[y0 : y0 + 22 * sy, x0 + 18 * sx : x0 + 21 * sx] = True
    mask[y0 + 9 * sy : y0 + 13 * sy, x0 : x0 + 21 * sx] = True
    amplitude = 0.72
    challenged[mask] = np.clip(challenged[mask] + amplitude, 0.0, 1.0)
    return challenged.astype(np.float32), mask, amplitude


def local_error(truth: FloatArray, estimate: FloatArray) -> FloatArray:
    diff = (truth.astype(np.float64) - estimate.astype(np.float64)) ** 2
    return np.asarray(np.sqrt(gaussian_filter(diff, sigma=R_ERROR_WINDOW_SIGMA, mode="nearest")), dtype=np.float32)


def band_limited(image: FloatArray, model: EstimatedCameraModel) -> FloatArray:
    band = supported_band(image.shape, model)
    return np.asarray(np.fft.ifft2(np.fft.fft2(image.astype(np.float64)) * band).real, dtype=np.float32)


def evidence_cutoff(model: EstimatedCameraModel, shape: tuple[int, int]) -> float:
    """Highest radial frequency (cycles/HR px) with |H| >= min_transfer along an axis."""
    transfer = transfer_magnitude(shape, model)
    fx = np.abs(np.fft.fftfreq(shape[1]))
    row = transfer[0, :]
    supported = fx[row >= np.float32(DEFAULT_MIN_TRANSFER)]
    return float(supported.max()) if supported.size else 0.0


def method_metrics(target: Target, estimate: FloatArray) -> dict[str, float]:
    return {"psnr_db": psnr(target.image, estimate), "rmse": rmse(target.image, estimate), **feature_scores(target.feature_mask, estimate)}


@dataclass(slots=True)
class CaseResult:
    family: str
    seed: int
    target: Target
    capture: CaptureSet
    bicubic: FloatArray
    shift_add: FloatArray
    single_frame: FloatArray
    map_image: FloatArray
    solver: SolverReport
    base_gate: GateResult
    provenance: UInt8Array
    confidence: FloatArray
    challenged: FloatArray
    injection_mask: np.ndarray
    amplitude: float
    challenge_gate: GateResult
    challenge_provenance: UInt8Array
    challenge_confidence: FloatArray
    metrics: dict[str, Any]


def run_case(family: str, cfg: DemoConfig) -> CaseResult:
    target = make_target(family, cfg.hr_shape, cfg.root_seed)
    capture = simulate_capture(target, cfg)
    model = estimated_model_for(cfg)
    thresholds = thresholds_for(cfg)

    bicubic = bicubic_baseline(capture)
    shift_add = shift_add_reconstruct(capture)
    single, _ = single_frame_deconvolution(capture, model, cfg)
    map_image, solver = map_reconstruct(
        capture, model, regularization=cfg.regularization, max_iterations=cfg.max_iterations
    )
    base_gate = consistency_gate(map_image, capture, model, thresholds)
    provenance, confidence = classify_provenance(capture, base_gate, proposal_hr=map_image, model=model)
    challenged, injection_mask, amplitude = inject_false_feature(map_image)
    challenge_gate = consistency_gate(challenged, capture, model, thresholds)
    challenge_provenance, challenge_confidence = classify_provenance(
        capture, challenge_gate, proposal_hr=challenged, model=model
    )

    truth_band = band_limited(target.image, model)
    correct = local_error(truth_band, map_image) <= np.float32(R_ERROR_TOLERANCE)
    correct_full = local_error(target.image, map_image) <= np.float32(R_ERROR_TOLERANCE)
    r_mask = provenance == int(ProvenanceClass.R)
    r_count = int(r_mask.sum())
    injected_r = int((challenge_provenance[injection_mask] >= int(ProvenanceClass.R)).sum())
    metrics: dict[str, Any] = {
        "family": family,
        "seed": cfg.root_seed,
        "methods": {
            "bicubic": method_metrics(target, bicubic),
            "single_frame_deconvolution": method_metrics(target, single),
            "shift_add": method_metrics(target, shift_add),
            "map": method_metrics(target, map_image),
        },
        "gate": {
            "base_pass_fraction": float(base_gate.pass_mask_hr.mean()),
            "challenge_detection_rate": challenge_detection_rate(
                injection_mask, base_gate.pass_mask_hr, challenge_gate.pass_mask_hr
            ),
            "challenge_amplitude": amplitude,
            "base_reasons": base_gate.reasons,
            "challenge_reasons": challenge_gate.reasons,
        },
        "provenance": {
            "counts": {ProvenanceClass(i).name: int((provenance == i).sum()) for i in range(4)},
            "r_fraction": r_count / provenance.size,
            "false_r_rate": float((r_mask & ~correct).sum() / r_count) if r_count else 0.0,
            "false_r_rate_full_band": float((r_mask & ~correct_full).sum() / r_count) if r_count else 0.0,
            "evidence_cutoff_cycles_per_hr_px": evidence_cutoff(model, map_image.shape),
            "verified_correct_fraction": float((r_mask & correct).sum() / provenance.size),
            "injected_pixels_labeled_r_or_m": injected_r,
            "m_count": int((provenance == int(ProvenanceClass.M)).sum()),
            "ece": expected_calibration_error(confidence.ravel(), correct.ravel().astype(np.float64)),
        },
        "solver": {"info": solver.info, "iterations": solver.iterations},
    }
    return CaseResult(
        family, cfg.root_seed, target, capture, bicubic, shift_add, single, map_image, solver, base_gate,
        provenance, confidence, challenged, injection_mask, amplitude, challenge_gate,
        challenge_provenance, challenge_confidence, metrics,
    )
