# UNFILED-IP
from __future__ import annotations

import numpy as np

from eczis.field.tiled_raster import TiledRasterRenderer
from eczis.provenance.accountant import demote
from eczis.schemas import ProvenanceClass


def test_renderer_never_promotes() -> None:
    image = np.ones((8, 8), dtype=np.float32) * 0.5
    provenance = np.full((8, 8), int(ProvenanceClass.R), dtype=np.uint8)
    confidence = np.ones((8, 8), dtype=np.float32)
    renderer = TiledRasterRenderer(image, provenance, confidence, scale=4)
    native = renderer.query(0.5, 0.5, 1.0)
    beyond = renderer.query(0.5, 0.5, 20.0)
    assert native.provenance == ProvenanceClass.R
    assert beyond.provenance == ProvenanceClass.U
    assert beyond.provenance <= native.provenance


def test_demotion_uses_conservative_minimum() -> None:
    base = np.array([[3, 2, 1, 0]], dtype=np.uint8)
    limiter = np.array([[2, 3, 0, 1]], dtype=np.uint8)
    assert np.array_equal(demote(base, limiter), np.array([[2, 2, 0, 0]], dtype=np.uint8))
