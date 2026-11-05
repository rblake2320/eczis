# UNFILED-IP
"""Gate and provenance attacks. These are the honesty layer: weakening any is wrong by definition."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from eczis.bench.pipeline import (
    CaseResult,
    estimated_model_for,
    local_error,
    run_case,
    simulate_capture,
    thresholds_for,
)
from eczis.config import DemoConfig
from eczis.field.coordinate_field import CoordinateFieldRenderer
from eczis.field.tiled_raster import TiledRasterRenderer
from eczis.gate.cycle import consistency_gate
from eczis.provenance.accountant import classify_provenance
from eczis.recon.map_solver import map_reconstruct
from eczis.schemas import ProvenanceClass
from eczis.targets.base import Target

R = int(ProvenanceClass.R)


def _contradict(frame_data: np.ndarray, cfg: DemoConfig, sigmas: float = 6.0) -> None:
    """Add a contradiction of ``sigmas`` local noise std (detectability defined in noise units)."""
    from eczis.camera.noise import predicted_variance

    block = frame_data[8:14, 10:18]
    std = np.sqrt(predicted_variance(block, shot_electrons=cfg.shot_electrons, read_noise_std=cfg.read_noise_std))
    frame_data[8:14, 10:18] = block + np.float32(sigmas) * std


def test_single_contradicting_frame_is_rejected(coin_case: CaseResult, small_cfg: DemoConfig) -> None:
    """K-1 attack: a candidate matching all frames but one must fail where that frame disagrees."""
    model, thresholds = estimated_model_for(small_cfg), thresholds_for(small_cfg)
    capture = copy.deepcopy(coin_case.capture)
    _contradict(capture.frames[4].data, small_cfg)
    scale = small_cfg.scale
    interior = np.zeros(coin_case.map_image.shape, dtype=np.bool_)
    interior[9 * scale : 13 * scale, 11 * scale : 17 * scale] = True

    gated = consistency_gate(coin_case.map_image, capture, model, thresholds)
    assert not gated.pass_mask_hr[interior].any()
    assert gated.reasons["contradicting_frame_count"] >= 1
    # Ablation proves the per-frame criterion is what closes the hole: the mean gate is diluted.
    diluted = consistency_gate(coin_case.map_image, capture, model, replace(thresholds, max_frame_z2_threshold=None))
    assert diluted.pass_mask_hr[interior].mean() > 0.99


def test_low_residual_nonidentifiable_detail_is_not_r(coin_case: CaseResult, small_cfg: DemoConfig) -> None:
    """Stop condition 6: detail invisible to the operator passes the gate but must never be R."""
    model, thresholds = estimated_model_for(small_cfg), thresholds_for(small_cfg)
    yy, xx = np.mgrid[: coin_case.map_image.shape[0], : coin_case.map_image.shape[1]]
    checker = np.where((yy + xx) % 2 == 0, 0.08, -0.08).astype(np.float32)
    region = np.zeros(coin_case.map_image.shape, dtype=np.bool_)
    region[30:70, 40:100] = True
    proposal = coin_case.map_image.copy()
    proposal[region] += checker[region]
    gated = consistency_gate(proposal, coin_case.capture, model, thresholds)
    assert gated.pass_mask_hr[region].mean() > 0.95, "precondition: residual is low (detail is invisible)"
    classes, _ = classify_provenance(coin_case.capture, gated, proposal_hr=proposal, model=model)
    assert not (classes[region] >= R).any()
    assert (coin_case.provenance[region] == R).any(), "precondition: region was R before the detail"


def test_noise_only_input_yields_no_verified_structure(small_cfg: DemoConfig) -> None:
    flat = Target("flat", np.full(small_cfg.hr_shape, 0.35, np.float32), np.zeros(small_cfg.hr_shape, np.bool_), 0)
    capture = simulate_capture(flat, small_cfg)
    model = estimated_model_for(small_cfg)
    recon, _ = map_reconstruct(capture, model, regularization=small_cfg.regularization, max_iterations=small_cfg.max_iterations)
    gated = consistency_gate(recon, capture, model, thresholds_for(small_cfg))
    classes, _ = classify_provenance(capture, gated, proposal_hr=recon, model=model)
    structure = local_error(flat.image, recon) > np.float32(0.08)
    assert not (structure & (classes == R)).any()


def test_false_r_bounded_across_three_snr_regimes(small_cfg: DemoConfig) -> None:
    """Registered limit: false-R <= 0.02 and zero injected R/M at 250, 700, 2500 e-."""
    for electrons in (250.0, 700.0, 2500.0):
        metrics = run_case("coin", replace(small_cfg, shot_electrons=electrons)).metrics
        assert metrics["provenance"]["false_r_rate"] <= 0.02, electrons
        assert metrics["provenance"]["injected_pixels_labeled_r_or_m"] == 0, electrons
        assert metrics["gate"]["base_pass_fraction"] >= 0.95, electrons


def test_prior_contribution_cannot_elevate_to_r(coin_case: CaseResult, small_cfg: DemoConfig) -> None:
    prior = np.ones(coin_case.map_image.shape, dtype=np.float32)
    classes, _ = classify_provenance(
        coin_case.capture, coin_case.base_gate, proposal_hr=coin_case.map_image,
        model=estimated_model_for(small_cfg), prior_contribution=prior,
    )
    assert not (classes >= R).any()


def test_empty_evidence_maps_entirely_to_u(coin_case: CaseResult, small_cfg: DemoConfig) -> None:
    capture = copy.deepcopy(coin_case.capture)
    for frame in capture.frames:
        frame.valid_mask[:] = False
    model = estimated_model_for(small_cfg)
    gated = consistency_gate(coin_case.map_image, capture, model, thresholds_for(small_cfg))
    classes, confidence = classify_provenance(capture, gated, proposal_hr=coin_case.map_image, model=model)
    assert (classes == int(ProvenanceClass.U)).all()
    assert float(confidence.max()) == 0.0


def test_contradictory_subset_is_never_r(coin_case: CaseResult, small_cfg: DemoConfig) -> None:
    capture = copy.deepcopy(coin_case.capture)
    for frame in capture.frames[::2]:
        frame.data[4:10, 4:12] = np.float32(0.95)
    model = estimated_model_for(small_cfg)
    gated = consistency_gate(coin_case.map_image, capture, model, thresholds_for(small_cfg))
    classes, _ = classify_provenance(capture, gated, proposal_hr=coin_case.map_image, model=model)
    s = small_cfg.scale
    assert not (classes[5 * s : 9 * s, 5 * s : 11 * s] >= R).any()


def test_pipeline_never_emits_m_for_any_family(small_cfg: DemoConfig) -> None:
    for family in ("microglyph", "siemens", "coin", "iris", "pcb"):
        case = run_case(family, small_cfg)
        assert case.metrics["provenance"]["m_count"] == 0, family
        assert case.metrics["provenance"]["injected_pixels_labeled_r_or_m"] == 0, family


_classes = st.integers(min_value=0, max_value=3)


@settings(max_examples=60, deadline=None)
@given(
    data=st.lists(_classes, min_size=64, max_size=64),
    x=st.floats(0.0, 1.0),
    y=st.floats(0.0, 1.0),
    scale=st.floats(0.25, 64.0),
)
def test_renderers_never_promote_property(data: list[int], x: float, y: float, scale: float) -> None:
    provenance = np.asarray(data, dtype=np.uint8).reshape(8, 8)
    rng = np.random.default_rng(len(data))
    image = rng.random((8, 8)).astype(np.float32)
    confidence = np.ones((8, 8), dtype=np.float32)
    iy, ix = int(round(y * 7)), int(round(x * 7))
    source = int(provenance[iy, ix])
    for renderer in (
        TiledRasterRenderer(image, provenance, confidence, scale=4),
        CoordinateFieldRenderer(image, provenance, confidence, scale=4, tile=4, features=8),
    ):
        result = renderer.query(x, y, scale)
        assert int(result.provenance) <= source
        if scale > 4 * 3.0:
            assert result.provenance == ProvenanceClass.U
        elif scale > 4 * 1.5:
            assert int(result.provenance) <= int(ProvenanceClass.I)
        assert 0.0 <= result.confidence <= 1.0


def test_gate_rejects_malformed_proposals(coin_case: CaseResult, small_cfg: DemoConfig) -> None:
    import pytest

    model, thresholds = estimated_model_for(small_cfg), thresholds_for(small_cfg)
    bad = coin_case.map_image.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        consistency_gate(bad, coin_case.capture, model, thresholds)
    with pytest.raises(ValueError, match="shape"):
        consistency_gate(coin_case.map_image[:-4], coin_case.capture, model, thresholds)


def test_k1_attack_sweep_over_frames(coin_case: CaseResult, small_cfg: DemoConfig) -> None:
    """Every single frame, when it alone contradicts, is caught (no privileged frame)."""
    model, thresholds = estimated_model_for(small_cfg), thresholds_for(small_cfg)
    s = small_cfg.scale
    for index in range(len(coin_case.capture.frames)):
        capture = copy.deepcopy(coin_case.capture)
        _contradict(capture.frames[index].data, small_cfg)
        gated = consistency_gate(coin_case.map_image, capture, model, thresholds)
        assert not gated.pass_mask_hr[9 * s : 13 * s, 11 * s : 17 * s].any(), index


