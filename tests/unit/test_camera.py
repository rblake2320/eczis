# UNFILED-IP
from __future__ import annotations

import numpy as np

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.camera.psf import gaussian_kernel
from eczis.camera.truth_model import TruthCameraModel
from eczis.schemas import ObservationParams


def _obs(seed: int = 1) -> ObservationParams:
    return ObservationParams("F", 0.25, 0.5, 0.01, 1.02, 0.03, -0.02, seed)


def test_truth_and_estimated_models_are_distinct() -> None:
    assert TruthCameraModel is not EstimatedCameraModel


def test_forward_adjoint_inner_product_identity() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(32, 40)).astype(np.float32)
    y = rng.normal(size=(16, 20)).astype(np.float32)
    model = EstimatedCameraModel(scale=2, psf_sigma_lr=0.7)
    lhs = float(np.vdot(model.project(x, _obs()), y))
    rhs = float(np.vdot(x, model.adjoint(y, _obs())))
    relative = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-9)
    assert relative < 1e-5


def test_psf_energy_conservation() -> None:
    size = 9
    kernel = gaussian_kernel(size, sigma=1.25)
    assert kernel.shape == (size, size)
    assert abs(float(kernel.sum()) - 1.0) < 1e-6


def test_truth_noise_is_seed_deterministic() -> None:
    scene = np.full((32, 40), 0.4, dtype=np.float32)
    model = TruthCameraModel(
        scale=2,
        psf_sigma_lr=0.8,
        read_noise_std=0.004,
        shot_electrons=700,
        saturation_level=1.0,
    )
    first = model.project(scene, _obs(seed=99))
    second = model.project(scene, _obs(seed=99))
    assert np.array_equal(first.data, second.data)
    assert np.array_equal(first.valid_mask, second.valid_mask)
