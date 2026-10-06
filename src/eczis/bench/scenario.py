# UNFILED-IP
"""Reusable synthetic scenario runner for validation-battery experiments.

Runs one target through capture -> MAP reconstruction -> cycle gate -> provenance
with independently configurable truth and estimated camera models, so that
challenges (null-space detail, model mismatch, SNR, threshold sweeps) can be
expressed without duplicating the demo harness.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from eczis.camera.calibration import calibration_from_config
from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.camera.truth_model import TruthCameraModel
from eczis.config import DemoConfig
from eczis.gate.cycle import consistency_gate
from eczis.gate.thresholds import GateThresholds
from eczis.provenance.accountant import classify_provenance
from eczis.recon.map_solver import map_reconstruct
from eczis.rig.schedule import lattice_schedule
from eczis.rig.simulated import SimulatedRig
from eczis.schemas import BoolArray, CaptureSet, FloatArray, GateResult, UInt8Array
from eczis.targets import (
    create_coin_target,
    create_iris_target,
    create_microglyph_target,
    create_pcb_target,
    create_siemens_target,
)
from eczis.targets.base import Target

TARGET_FAMILIES = ("microglyph", "siemens", "pcb", "coin", "iris")


def make_target(family: str, cfg: DemoConfig) -> Target:
    shape = cfg.hr_shape
    if family == "microglyph":
        return create_microglyph_target(shape, seed=cfg.root_seed, stroke_px=1)
    if family == "siemens":
        return create_siemens_target(shape)
    if family == "pcb":
        return create_pcb_target(shape)
    if family == "coin":
        return create_coin_target(shape)
    if family == "iris":
        return create_iris_target(shape)
    raise ValueError(f"unknown target family: {family}")


@dataclass(frozen=True, slots=True)
class Scenario:
    cfg: DemoConfig
    target: Target
    capture: CaptureSet
    model: EstimatedCameraModel
    thresholds: GateThresholds
    reconstruction: FloatArray
    gate: GateResult
    provenance: UInt8Array
    confidence: FloatArray

    def evaluate(self, proposal: FloatArray) -> tuple[GateResult, UInt8Array, FloatArray]:
        gate = consistency_gate(proposal, self.capture, self.model, self.thresholds)
        classes, confidence = classify_provenance(self.capture, gate)
        return gate, classes, confidence


def run_scenario(family: str = "microglyph", cfg: DemoConfig | None = None) -> Scenario:
    cfg = cfg or DemoConfig(max_iterations=12)
    target = make_target(family, cfg)
    truth = TruthCameraModel(
        scale=cfg.scale,
        psf_sigma_lr=cfg.psf_sigma_lr_truth,
        read_noise_std=cfg.read_noise_std,
        shot_electrons=cfg.shot_electrons,
        saturation_level=cfg.saturation_level,
    )
    model = EstimatedCameraModel(scale=cfg.scale, psf_sigma_lr=cfg.psf_sigma_lr_estimated)
    capture = SimulatedRig(truth, target, calibration_from_config(cfg)).capture_burst(
        lattice_schedule(cfg.frame_count, cfg.root_seed), root_seed=cfg.root_seed
    )
    recon, _ = map_reconstruct(
        capture, model, regularization=cfg.regularization, max_iterations=cfg.max_iterations
    )
    thresholds = GateThresholds(cfg.gate_z2_threshold, min_valid_frames=max(4, cfg.frame_count // 4))
    gate = consistency_gate(recon, capture, model, thresholds)
    classes, confidence = classify_provenance(capture, gate)
    return Scenario(cfg, target, capture, model, thresholds, recon, gate, classes, confidence)


def with_config(cfg: DemoConfig | None = None, **changes: object) -> DemoConfig:
    return replace(cfg or DemoConfig(max_iterations=12), **changes)  # type: ignore[arg-type]


def null_space_detail(
    shape: tuple[int, int], center: tuple[int, int], *, amplitude: float, radius: float
) -> tuple[FloatArray, BoolArray]:
    """A Nyquist-frequency checkerboard under a smooth envelope.

    Block averaging with an even scale annihilates the checkerboard exactly and the
    estimated PSF attenuates its residual envelope leakage below float32 noise, so
    A h ~= 0: the detail is invisible to every observation.
    """
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = center
    envelope = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * radius**2))
    checker = np.where((yy + xx) % 2 == 0, 1.0, -1.0)
    detail = (amplitude * envelope * checker).astype(np.float32)
    mask = envelope >= 0.5
    return detail, mask.astype(np.bool_)


def high_passed_glyph(
    shape: tuple[int, int], center: tuple[int, int], *, amplitude: float, model: EstimatedCameraModel
) -> tuple[FloatArray, BoolArray]:
    """A plausible fake 'H' glyph with its observable content projected out.

    h_null = g - P(g), where P is a low-pass matched to the estimated PSF passband;
    this mimics hallucinated high-frequency texture from a generative upscaler.
    """
    h, w = shape
    glyph = np.zeros(shape, dtype=np.float32)
    cy, cx = center
    glyph[cy - 6 : cy + 6, cx - 5 : cx - 3] = 1.0
    glyph[cy - 6 : cy + 6, cx + 3 : cx + 5] = 1.0
    glyph[cy - 1 : cy + 1, cx - 5 : cx + 5] = 1.0
    sigma_hr = model.psf_sigma_lr * model.scale
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    passband = np.exp(-2.0 * (np.pi**2) * (sigma_hr * 0.6) ** 2 * (fx**2 + fy**2))
    observable = np.fft.ifft2(np.fft.fft2(glyph) * passband).real
    detail = (amplitude * (glyph - observable)).astype(np.float32)
    return detail, glyph > 0
