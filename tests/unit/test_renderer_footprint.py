# UNFILED-IP
"""Renderer provenance must cover every sample that contributes to a rendered value.

Originally a strict xfail (validation battery, PR #7): the class came from the nearest pixel
while the value blended 4 neighbours. Repaired by eczis.field.policy.bilinear_footprint.
"""
from __future__ import annotations

import numpy as np

from eczis.field.tiled_raster import TiledRasterRenderer
from eczis.schemas import ProvenanceClass


def _split_renderer() -> TiledRasterRenderer:
    image = np.zeros((4, 4), dtype=np.float32)
    image[:, 2:] = 1.0
    provenance = np.full((4, 4), int(ProvenanceClass.R), dtype=np.uint8)
    provenance[:, 2:] = int(ProvenanceClass.U)
    confidence = np.ones((4, 4), dtype=np.float32)
    return TiledRasterRenderer(image, provenance, confidence, scale=4)


def test_query_between_classes_blends_values_from_both() -> None:
    # x=0.45 -> px=1.35: bilinear mixes column 1 (R, value 0) and column 2 (U, value 1).
    result = _split_renderer().query(0.45, 0.5, 1.0)
    assert 0.0 < result.value < 1.0


def test_blended_value_carries_weakest_contributing_class() -> None:
    result = _split_renderer().query(0.45, 0.5, 1.0)
    assert result.provenance == ProvenanceClass.U


def test_integer_query_ignores_zero_weight_neighbours() -> None:
    r = _split_renderer()
    # px=1 exactly: column 2 (U) has zero weight, so the R pixel alone decides.
    result = r.query_pixel(1.0, 1.0, 4.0)
    assert result.provenance == ProvenanceClass.R
    assert result.value == 0.0


def test_vertical_and_diagonal_mixed_edges() -> None:
    image = np.zeros((4, 4), dtype=np.float32)
    prov = np.full((4, 4), int(ProvenanceClass.R), dtype=np.uint8)
    prov[2, 2] = int(ProvenanceClass.I)
    r = TiledRasterRenderer(image, prov, np.ones((4, 4), np.float32), scale=4)
    assert r.query_pixel(1.5, 1.5, 4.0).provenance == ProvenanceClass.I  # diagonal contributor
    assert r.query_pixel(2.0, 1.5, 4.0).provenance == ProvenanceClass.I  # vertical pair
    assert r.query_pixel(1.0, 1.5, 4.0).provenance == ProvenanceClass.R  # (2,2) not in footprint


def test_boundary_and_out_of_range_queries_clamp() -> None:
    r = _split_renderer()
    assert r.query_pixel(3.0, 3.0, 4.0).provenance == ProvenanceClass.U
    assert r.query_pixel(99.0, -5.0, 4.0).provenance == ProvenanceClass.U
    assert r.query_pixel(-5.0, 99.0, 4.0).provenance == ProvenanceClass.R


def test_scale_policy_thresholds_and_zoom_out_restores_native() -> None:
    r = _split_renderer()
    assert r.query_pixel(0.0, 0.0, 4.0 * 1.5).provenance == ProvenanceClass.R
    assert r.query_pixel(0.0, 0.0, 4.0 * 1.51).provenance == ProvenanceClass.I
    assert r.query_pixel(0.0, 0.0, 4.0 * 3.01).provenance == ProvenanceClass.U
    assert r.query_pixel(0.0, 0.0, 4.0).provenance == ProvenanceClass.R
    assert int(r.provenance[0, 0]) == int(ProvenanceClass.R)  # stored class never altered
