# UNFILED-IP
"""Property-based checks of the provenance algebra and renderer, plus degenerate inputs."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from eczis.bench.scenario import run_scenario
from eczis.field.tiled_raster import TiledRasterRenderer
from eczis.provenance.accountant import demote
from eczis.schemas import ProvenanceClass

classes = st.lists(st.integers(0, 3), min_size=16, max_size=16)


@given(classes, classes, classes)
def test_demote_is_monotone_commutative_associative(a: list[int], b: list[int], c: list[int]) -> None:
    x, y, z = (np.array(v, dtype=np.uint8).reshape(4, 4) for v in (a, b, c))
    assert np.array_equal(demote(x, y), demote(y, x))
    assert np.array_equal(demote(demote(x, y), z), demote(x, demote(y, z)))
    assert (demote(x, y) <= x).all() and (demote(x, y) <= y).all()
    assert np.array_equal(demote(x, x), x)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.integers(0, 3), min_size=64, max_size=64),
    st.floats(-0.5, 1.5),
    st.floats(-0.5, 1.5),
    st.floats(1e-3, 64.0),
)
def test_renderer_never_exceeds_base_class_and_zoom_never_promotes(
    prov: list[int], x: float, y: float, scale: float
) -> None:
    p = np.array(prov, dtype=np.uint8).reshape(8, 8)
    r = TiledRasterRenderer(np.zeros((8, 8), np.float32), p, np.ones((8, 8), np.float32), scale=4)
    out = r.query(x, y, scale)
    assert int(out.provenance) <= int(p.max())
    assert int(r.query(x, y, scale * 2).provenance) <= int(out.provenance)
    assert 0.0 <= out.confidence <= 1.0


def test_nan_proposal_never_yields_reconstructed() -> None:
    s = run_scenario("microglyph")
    bad = s.reconstruction.copy()
    bad[40:60, 40:60] = np.nan
    gate, classes_, _ = s.evaluate(bad)
    assert not gate.pass_mask_hr[40:60, 40:60].any()
    assert int((classes_[40:60, 40:60] >= ProvenanceClass.R).sum()) == 0


def test_shape_mismatches_raise() -> None:
    with pytest.raises(ValueError):
        demote(np.zeros((2, 2), np.uint8), np.zeros((3, 3), np.uint8))
    with pytest.raises(ValueError):
        TiledRasterRenderer(np.zeros((2, 2), np.float32), np.zeros((3, 3), np.uint8), np.zeros((2, 2), np.float32), 4)
