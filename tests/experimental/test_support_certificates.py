# UNFILED-IP
"""Validation of the EXPERIMENTAL support field and answer certificates (research, issue #10).

Pre-registered expectations (2026-10-06, written before these tests were run):
  D1 coarse bands are supported (R) and the finest band is unsupported under the base burst;
     orthonormal atoms make the comparison energy-matched by construction.
  D2 a proposal carrying hallucinated fine detail cannot earn R in the bands that carry it, even
     though it reprojects within the old residual gate (which labels it R).
  D3 posterior alternatives vary far more in unsupported bands than in supported ones.
  D4 "is the left stroke present?" (8 vs 3) is UNDETERMINED for both glyphs under the base burst.
  D5 a closer capture certifies the correct answer for both glyphs, the interval contains the
     known truth, and a redundant capture does not certify; the ranking puts informative first.
  D6 a pixel class map derived from the support field can feed the renderer without breaking
     never-promote / zoom monotonicity.
"""
from __future__ import annotations

import numpy as np
import pytest

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.experimental.certificates import Decision, Question, best_next_measurement, certify
from eczis.experimental.support import (
    Measurement,
    SupportField,
    build_operator,
    noise_variance,
    posterior_samples,
    support_field,
)
from eczis.field.tiled_raster import TiledRasterRenderer
from eczis.rig.schedule import lattice_schedule
from eczis.schemas import ObservationParams, ProvenanceClass

N = 32
SHOT, READ = 700.0, 0.004


def _glyph(eight: bool, c: float = 0.5) -> np.ndarray:
    x = np.full((N, N), 0.2)
    x[8:10, 10:22] += c
    x[16:18, 10:22] += c
    x[24:26, 10:22] += c
    x[8:26, 20:22] += c
    if eight:
        x[8:26, 10:12] += c
    return np.clip(x, 0.0, 1.0)


def _stroke_question() -> Question:
    m = np.zeros((N, N))
    m[11:23, 10:12] = 1.0
    return Question("left_stroke_present", (m / m.sum()).ravel(), 0.45)


def _capture(scale: int, psf: float, phases: list[tuple[float, float]], tag: str) -> list[Measurement]:
    model = EstimatedCameraModel(scale=scale, psf_sigma_lr=psf)
    return [Measurement(model, ObservationParams(f"{tag}{i}", dx, dy, 0.0, 1.0, 0.0, 0.0, 500 + i))
            for i, (dx, dy) in enumerate(phases)]


@pytest.fixture(scope="module")
def operators() -> dict[str, np.ndarray]:
    base = [Measurement(EstimatedCameraModel(scale=4, psf_sigma_lr=0.70), o)
            for o in lattice_schedule(16, 20260803).observations]
    a = build_operator(base, (N, N))
    closer = build_operator(_capture(2, 0.5, [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)], "C"), (N, N))
    macro = build_operator(_capture(1, 0.6, [(0, 0), (0.5, 0.5)], "M"), (N, N))
    return {"base": a, "closer": closer, "macro": macro, "redundant": a[: 4 * 64]}


def _observe(a: np.ndarray, truth: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y = a @ truth.ravel()
    var = noise_variance(y, SHOT, READ)
    return y + np.random.default_rng(seed).normal(0.0, np.sqrt(var)), var


@pytest.fixture(scope="module")
def base_field(operators: dict[str, np.ndarray]) -> SupportField:
    y, var = _observe(operators["base"], _glyph(True), 11)
    return support_field(operators["base"], y, var, (N, N))


def test_d1_coarse_bands_supported_finest_unsupported(base_field: SupportField) -> None:
    cls, lv = base_field.coefficient_class, base_field.level
    assert np.all(cls[lv >= 4] == int(ProvenanceClass.R))
    assert np.all(cls[lv == 1] == int(ProvenanceClass.U))
    assert np.median(base_field.amplitude[lv == 1]) > 10 * np.median(base_field.amplitude[lv == 4])


def test_d2_hallucinated_detail_cannot_earn_r_by_reprojection(
    operators: dict[str, np.ndarray], base_field: SupportField
) -> None:
    from eczis.bench.scenario import null_space_detail

    detail, mask = null_space_detail((N, N), (16, 16), amplitude=0.3, radius=4)
    a = operators["base"]
    proposal = base_field.mean + detail.ravel()
    # It reprojects essentially identically (the old gate's criterion).
    assert np.max(np.abs(a @ detail.ravel())) < 1e-4
    coeff = base_field.evaluate_proposal(proposal)
    fine_view = base_field.pixel_class(1, coeff_class=coeff)
    assert int((fine_view[mask] >= int(ProvenanceClass.R)).sum()) == 0


def test_d3_alternatives_vary_in_unsupported_bands(base_field: SupportField) -> None:
    samples = posterior_samples(base_field, 64, seed=3)
    coeffs = samples @ base_field.basis.T
    spread = coeffs.std(axis=0) / (2.0 ** np.minimum(base_field.level, 5))
    lv = base_field.level
    assert np.median(spread[lv == 1]) > 10 * np.median(spread[lv >= 4])


@pytest.mark.parametrize("eight", [True, False])
def test_d4_glyph_question_undetermined_under_base_burst(operators: dict[str, np.ndarray], eight: bool) -> None:
    y, var = _observe(operators["base"], _glyph(eight), 3 if eight else 4)
    cert = certify(support_field(operators["base"], y, var, (N, N)), _stroke_question())
    assert cert.decision == Decision.UNDETERMINED


@pytest.mark.parametrize("eight", [True, False])
def test_d5_informative_capture_certifies_redundant_does_not(operators: dict[str, np.ndarray], eight: bool) -> None:
    q = _stroke_question()
    truth = _glyph(eight)
    true_answer = float(q.weights @ truth.ravel())
    assert true_answer == pytest.approx(0.75 if eight else 0.2833, abs=1e-3)  # frozen ground truth
    results = {}
    for extra in ("closer", "macro", "redundant"):
        a2 = np.vstack([operators["base"], operators[extra]])
        y, var = _observe(a2, truth, 21 if eight else 22)
        results[extra] = certify(support_field(a2, y, var, (N, N)), q)
    for informative in ("closer", "macro"):
        cert = results[informative]
        assert cert.decision == (Decision.SUPPORTED_ABOVE if eight else Decision.SUPPORTED_BELOW), informative
        assert cert.answer == ("present" if eight else "absent")
        assert cert.lower <= true_answer <= cert.upper, informative
        # the three interval mechanisms agree: 200 posterior samples stay within +/-4 sigma of
        # the estimate (a few may pass +/-3 sigma by chance), and the null push hits +/-3 sigma.
        sd = (cert.upper - cert.lower) / 6.0
        assert cert.lower - sd <= cert.sample_range[0] and cert.sample_range[1] <= cert.upper + sd
        assert cert.null_push_range[0] == pytest.approx(cert.lower) and cert.null_push_range[1] == pytest.approx(cert.upper)
    assert results["redundant"].decision == Decision.UNDETERMINED
    y0, v0 = _observe(operators["base"], truth, 5)
    prec = np.linalg.inv(support_field(operators["base"], y0, v0, (N, N)).cov)
    ranked = best_next_measurement(
        prec, [(k, operators[k], noise_variance(operators[k] @ truth.ravel(), SHOT, READ))
               for k in ("redundant", "closer", "macro")], q.weights)
    assert [name for name, _ in ranked] == ["macro", "closer", "redundant"]


def test_d6_support_class_map_feeds_renderer_without_promotion(base_field: SupportField) -> None:
    for view_level in range(1, int(base_field.level.max()) + 1):
        cls_map = base_field.pixel_class(view_level)
        if view_level > 1:
            assert np.all(cls_map >= base_field.pixel_class(view_level - 1))  # finer view never stronger
        r = TiledRasterRenderer(base_field.mean.reshape(N, N).astype(np.float32), cls_map,
                                np.ones((N, N), np.float32), scale=4)
        for x, y in [(0.0, 0.0), (0.37, 0.61), (1.0, 1.0)]:
            q1, q2 = r.query(x, y, 4.0), r.query(x, y, 16.0)
            assert int(q1.provenance) <= int(cls_map.max())
            assert int(q2.provenance) <= int(q1.provenance)


def test_d6b_weakest_contributor_when_stacked_on_renderer_repair(base_field: SupportField) -> None:
    pytest.importorskip("eczis.field.policy", reason="runs when combined with PR #9")
    cls_map = base_field.pixel_class(2)
    r = TiledRasterRenderer(base_field.mean.reshape(N, N).astype(np.float32), cls_map,
                            np.ones((N, N), np.float32), scale=4)
    for px in np.linspace(0, N - 1, 23):
        res = r.query_pixel(float(px), 7.3, 4.0)  # type: ignore[attr-defined]
        x0 = int(np.floor(px))
        neighbours = cls_map[7:9, x0 : min(x0 + 2, N)]
        assert int(res.provenance) <= int(neighbours.min()) or float(px).is_integer()


def test_action_ranking_matches_observed_uncertainty_reduction(operators: dict[str, np.ndarray]) -> None:
    q = _stroke_question()
    truth = _glyph(True)
    y0, v0 = _observe(operators["base"], truth, 5)
    sf0 = support_field(operators["base"], y0, v0, (N, N))
    base_sd = float(np.sqrt(q.weights @ sf0.cov @ q.weights))
    predicted = dict(best_next_measurement(
        np.linalg.inv(sf0.cov),
        [(k, operators[k], noise_variance(operators[k] @ truth.ravel(), SHOT, READ))
         for k in ("redundant", "closer", "macro")], q.weights))
    observed = {}
    for k in ("redundant", "closer", "macro"):
        a2 = np.vstack([operators["base"], operators[k]])
        y, var = _observe(a2, truth, 31)
        sf = support_field(a2, y, var, (N, N))
        observed[k] = float(np.sqrt(q.weights @ sf.cov @ q.weights))
        assert observed[k] == pytest.approx(predicted[k], rel=0.05), k  # prediction matches outcome
    gain = {k: 1 - observed[k] / base_sd for k in observed}
    assert gain["macro"] > gain["closer"] > gain["redundant"] >= 0.0
    assert gain["macro"] > 0.9 and gain["closer"] > 0.5
    # Fresh-noise repeats may average noise down a little; they must not act like new geometry.
    assert gain["redundant"] < 0.05


def test_redundant_repeats_cannot_shrink_unseen_directions(operators: dict[str, np.ndarray]) -> None:
    """Repeating the same geometry reduces noise only inside the row space of A.

    Exact-zero gain applies to derived data (a deterministic function of data already held, such
    as a digital enlargement); independent-noise repeats legitimately average noise.
    """
    a = operators["base"]
    y, v = _observe(a, _glyph(True), 41)
    sf1 = support_field(a, y, v, (N, N))
    a2 = np.vstack([a, a, a, a])
    y2, v2 = _observe(a2, _glyph(True), 42)
    sf4 = support_field(a2, y2, v2, (N, N))
    lv = sf1.level
    fine_1, fine_4 = np.median(sf1.amplitude[lv == 1]), np.median(sf4.amplitude[lv == 1])
    assert fine_4 > 0.9 * fine_1  # 4x the frames barely touch the unseen finest band
    assert np.all(sf4.coefficient_class[lv == 1] == int(ProvenanceClass.U))
    assert certify(sf4, _stroke_question()).decision == Decision.UNDETERMINED


def test_frozen_band_support_values(base_field: SupportField) -> None:
    """Freeze intermediate quantities so a threshold change cannot silently preserve labels."""
    med = {lv: float(np.median(base_field.amplitude[base_field.level == lv])) for lv in (1, 2, 3, 4, 5)}
    assert med[1] == pytest.approx(0.496, abs=0.02)
    assert med[2] == pytest.approx(0.23, abs=0.02)
    assert med[3] == pytest.approx(0.072, abs=0.01)
    assert med[4] == pytest.approx(0.025, abs=0.005)
    assert med[5] == pytest.approx(0.008, abs=0.003)
