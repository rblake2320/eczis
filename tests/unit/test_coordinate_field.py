# UNFILED-IP
from __future__ import annotations

import numpy as np

from eczis.field.coordinate_field import CoordinateFieldRenderer
from eczis.field.tiled_raster import TiledRasterRenderer
from eczis.schemas import ProvenanceClass


def _smooth(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return np.asarray(0.5 + 0.2 * np.sin(xx / 9.0) * np.cos(yy / 11.0), dtype=np.float32)


def test_field_reproduces_source_grid_within_tolerance() -> None:
    image = _smooth((32, 32))
    prov = np.full(image.shape, int(ProvenanceClass.R), np.uint8)
    field = CoordinateFieldRenderer(image, prov, np.ones_like(image), scale=4, tile=8, features=32)
    errors = []
    for y in range(0, 32, 3):
        for x in range(0, 32, 3):
            errors.append(abs(field.query(x / 31, y / 31, 1.0).value - float(image[y, x])))
    assert max(errors) < 0.02
    assert field.query(0.5, 0.5, 1.0).provenance == ProvenanceClass.R


def test_poor_fit_tiles_are_capped_at_inferred() -> None:
    rng = np.random.default_rng(3)
    image = rng.random((16, 16)).astype(np.float32)  # white noise: unfittable with few features
    prov = np.full(image.shape, int(ProvenanceClass.R), np.uint8)
    field = CoordinateFieldRenderer(image, prov, np.ones_like(image), scale=4, tile=8, features=8)
    assert field.fit_rmse(0.2, 0.2) > 0.02
    assert field.query(0.2, 0.2, 1.0).provenance == ProvenanceClass.I


def test_field_and_raster_agree_on_class_and_are_deterministic() -> None:
    image = _smooth((24, 24))
    prov = np.random.default_rng(1).integers(0, 3, image.shape).astype(np.uint8)
    conf = np.ones_like(image)
    a = CoordinateFieldRenderer(image, prov, conf, scale=4, tile=8, seed=7)
    b = CoordinateFieldRenderer(image, prov, conf, scale=4, tile=8, seed=7)
    raster = TiledRasterRenderer(image, prov, conf, scale=4)
    for x, y in ((0.1, 0.9), (0.5, 0.5), (0.77, 0.31)):
        assert a.query(x, y, 2.0) == b.query(x, y, 2.0)
        assert int(a.query(x, y, 2.0).provenance) <= int(raster.query(x, y, 2.0).provenance)
