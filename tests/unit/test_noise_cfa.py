# UNFILED-IP
from __future__ import annotations

import numpy as np
import pytest

from eczis.camera.cfa import CFA_PATTERNS, channel_masks, mosaic
from eczis.camera.noise import predicted_variance, sample_poisson_gaussian


@pytest.mark.parametrize("level", [0.05, 0.3, 0.8])
def test_poisson_gaussian_statistics_match_predicted_variance(level: float) -> None:
    signal = np.full((400, 400), level, dtype=np.float32)
    rng = np.random.default_rng(11)
    noisy = sample_poisson_gaussian(signal, rng, shot_electrons=700.0, read_noise_std=0.004)
    expected = float(predicted_variance(signal[:1, :1], shot_electrons=700.0, read_noise_std=0.004)[0, 0])
    assert abs(float(noisy.mean()) - level) < 3e-3
    assert abs(float(noisy.var()) / expected - 1.0) < 0.03


def test_noise_parameters_validated() -> None:
    with pytest.raises(ValueError):
        predicted_variance(np.zeros((2, 2), np.float32), shot_electrons=0.0, read_noise_std=0.0)
    with pytest.raises(ValueError):
        predicted_variance(np.zeros((2, 2), np.float32), shot_electrons=1.0, read_noise_std=-1.0)


def test_cfa_sample_locations_rggb() -> None:
    masks = channel_masks((4, 4), "RGGB")
    assert masks[0, 0, 0] and masks[1, 0, 1] and masks[1, 1, 0] and masks[2, 1, 1]
    assert masks[0].sum() == 4 and masks[1].sum() == 8 and masks[2].sum() == 4


@pytest.mark.parametrize("pattern", [p for p in CFA_PATTERNS if p != "MONO"])
def test_every_pixel_samples_exactly_one_channel(pattern: str) -> None:
    masks = channel_masks((6, 8), pattern)
    assert (masks.sum(axis=0) == 1).all()


def test_mosaic_reads_the_declared_channel() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.1, 0.5, 0.9
    assert np.allclose(mosaic(rgb, "BGGR"), [[0.9, 0.5], [0.5, 0.1]])


def test_unknown_cfa_pattern_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown CFA"):
        channel_masks((2, 2), "RGBW")
