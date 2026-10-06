# UNFILED-IP
"""Renderer provenance must cover every sample that contributes to a rendered value.

Pre-registered expectation: query() blends 4 neighbours (bilinear) but reads the class
from the single nearest pixel, so a value partly built from U is labelled R.
"""
from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.xfail(strict=True, reason="KNOWN GAP: class taken from nearest pixel, not the blend footprint")
def test_blended_value_carries_weakest_contributing_class() -> None:
    result = _split_renderer().query(0.45, 0.5, 1.0)
    assert result.provenance == ProvenanceClass.U
