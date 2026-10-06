# UNFILED-IP
"""EXPERIMENTAL: location-and-frequency support field from exact posterior uncertainty.

Small-patch prototype. The estimated forward operator A is built explicitly, and the Gaussian
posterior over the high-resolution patch is computed exactly:

    C = (A^T D A + I / s_p^2)^-1,   mu = C (A^T D y + m_p / s_p^2)

with D the inverse noise variance and a deliberately weak prior (s_p = 1 on a [0, 1] intensity
scale). Expressed in an orthonormal Haar basis, each coefficient k (location x band) gets the
per-pixel intensity uncertainty it contributes, sigma_post_k / 2**level. A coefficient the frames
pin down contributes little; one the camera cannot see stays prior-limited whatever its value. Unlike a per-atom
detectability score, the posterior covariance accounts for confounding between components.

Support is a property of the measurements, not of any proposal: a proposal earns R in a band only
if that band is supported AND the proposal agrees with the posterior there.

Class thresholds are research policy constants, not calibrated probabilities.
Not used by production classification.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.schemas import ObservationParams, ProvenanceClass

F64 = npt.NDArray[np.float64]

# Support is judged in intensity units: the posterior std of a coefficient's contribution to each
# pixel it covers (sigma_k / 2**level for orthonormal Haar), on a [0, 1] intensity scale.
R_MAX_AMPLITUDE = 0.05  # per-pixel uncertainty at or below this -> R
I_MAX_AMPLITUDE = 0.15  # at or below this -> I; above -> U


@dataclass(frozen=True, slots=True)
class Measurement:
    """One observation: the camera model that produced it plus its parameters."""

    model: EstimatedCameraModel
    params: ObservationParams


def build_operator(measurements: Sequence[Measurement], hr_shape: tuple[int, int]) -> F64:
    """Stack each measurement's linear projection of every unit HR pixel into a dense matrix."""
    n = hr_shape[0] * hr_shape[1]
    blocks = []
    for meas in measurements:
        cols = []
        for j in range(n):
            e = np.zeros(n, dtype=np.float32)
            e[j] = 1.0
            cols.append(meas.model.project(e.reshape(hr_shape), meas.params).ravel().astype(np.float64))
        blocks.append(np.stack(cols, axis=1))
    return np.vstack(blocks)


def haar_matrix(n: int) -> tuple[F64, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Orthonormal 2D Haar basis W (coeff = W @ x) for an n x n patch, n a power of two.

    Returns W, the band level of each coefficient (1 = finest detail, J+1 = coarse approximation)
    and, for each coefficient, the flat indices' block size 2**level it spans.
    """
    if n & (n - 1) or n < 2:
        raise ValueError("patch side must be a power of two")
    levels = int(np.log2(n))

    def forward(img: F64) -> F64:
        out = img.copy()
        size = n
        for _ in range(levels):
            block = out[:size, :size].copy()
            a = (block[:, 0::2] + block[:, 1::2]) / np.sqrt(2.0)
            d = (block[:, 0::2] - block[:, 1::2]) / np.sqrt(2.0)
            block = np.hstack([a, d])
            a = (block[0::2, :] + block[1::2, :]) / np.sqrt(2.0)
            d = (block[0::2, :] - block[1::2, :]) / np.sqrt(2.0)
            out[:size, :size] = np.vstack([a, d])
            size //= 2
        return out

    eye = np.eye(n * n)
    w = np.stack([forward(eye[j].reshape(n, n)).ravel() for j in range(n * n)], axis=1)
    level = np.zeros((n, n), dtype=np.int64)
    size = n
    for j in range(1, levels + 1):
        half = size // 2
        level[:size, :size] = j
        size = half
    level[:1, :1] = levels + 1
    return w, level.ravel(), (2 ** np.minimum(level, levels)).ravel()


def coefficient_footprints(n: int) -> list[npt.NDArray[np.int64]]:
    """Flat HR pixel indices each Haar coefficient spans (its spatial support)."""
    w, _, _ = haar_matrix(n)
    return [np.flatnonzero(np.abs(w[k]) > 1e-12) for k in range(n * n)]


@dataclass(frozen=True, slots=True)
class SupportField:
    shape: tuple[int, int]
    mean: F64  # posterior mean, flat HR
    cov: F64  # posterior covariance, HR pixel basis
    basis: F64  # Haar W
    level: npt.NDArray[np.int64]
    ratio: F64  # sigma_post / sigma_prior per coefficient (diagnostic)
    amplitude: F64  # per-pixel intensity uncertainty contributed by each coefficient
    prior_std: float
    footprints: list[npt.NDArray[np.int64]]

    @property
    def coefficient_class(self) -> npt.NDArray[np.uint8]:
        cls = np.full(self.ratio.shape, int(ProvenanceClass.U), dtype=np.uint8)
        cls[self.amplitude <= I_MAX_AMPLITUDE] = int(ProvenanceClass.I)
        cls[self.amplitude <= R_MAX_AMPLITUDE] = int(ProvenanceClass.R)
        return cls

    def coefficient_std(self) -> F64:
        return np.asarray(np.sqrt(np.einsum("ki,ij,kj->k", self.basis, self.cov, self.basis)), dtype=np.float64)

    def pixel_class(self, coarsest_excluded_below: int = 1, coeff_class: npt.NDArray[np.uint8] | None = None) -> npt.NDArray[np.uint8]:
        """Per-pixel class viewing bands with level >= ``coarsest_excluded_below``.

        Level 1 includes every band (full reconstruction-grid detail). A pixel takes the weakest
        class of every included coefficient whose spatial support covers it.
        """
        cls = self.coefficient_class if coeff_class is None else coeff_class
        out = np.full(self.mean.shape, int(ProvenanceClass.M), dtype=np.uint8)
        for k, fp in enumerate(self.footprints):
            if self.level[k] >= coarsest_excluded_below:
                out[fp] = np.minimum(out[fp], cls[k])
        return out.reshape(self.shape)

    def supported_level(self) -> npt.NDArray[np.int64]:
        """Finest band level at which each pixel is R (a per-location resolution limit)."""
        top = int(self.level.max())
        result = np.full(self.mean.shape, top + 1, dtype=np.int64)
        for lv in range(top, 0, -1):
            ok = self.pixel_class(lv).ravel() >= int(ProvenanceClass.R)
            result[ok] = lv
        return result.reshape(self.shape)

    def evaluate_proposal(self, proposal: F64, kappa: float = 3.0) -> npt.NDArray[np.uint8]:
        """Coefficient classes for a candidate image (e.g. a generative upscale).

        A coefficient is R only if its band is supported AND the proposal agrees with the
        posterior within kappa sigma; a supported band the proposal contradicts becomes U.
        Unsupported bands stay I/U regardless of how well the whole proposal reprojects.
        """
        base = self.coefficient_class.copy()
        diff = np.abs(self.basis @ (proposal.ravel() - self.mean))
        contradicted = (base == int(ProvenanceClass.R)) & (diff > kappa * self.coefficient_std())
        base[contradicted] = int(ProvenanceClass.U)
        return base


def support_field(
    a: F64,
    y: F64,
    noise_var: F64,
    shape: tuple[int, int],
    *,
    prior_std: float = 1.0,
    prior_mean: float | None = None,
) -> SupportField:
    if shape[0] != shape[1]:
        raise ValueError("prototype requires a square patch")
    d = 1.0 / np.maximum(noise_var, 1e-12)
    m_p = float(np.mean(y)) if prior_mean is None else prior_mean
    precision = (a.T * d) @ a + np.eye(a.shape[1]) / prior_std**2
    cov = np.linalg.inv(precision)
    cov = 0.5 * (cov + cov.T)
    mean = cov @ (a.T @ (d * y) + m_p / prior_std**2)
    w, level, span = haar_matrix(shape[0])
    coeff_std = np.sqrt(np.maximum(np.einsum("ki,ij,kj->k", w, cov, w), 0.0))
    amplitude = coeff_std / span.astype(np.float64)
    return SupportField(shape, mean, cov, w, level, coeff_std / prior_std, amplitude, prior_std,
                        coefficient_footprints(shape[0]))


def noise_variance(y: F64, shot_electrons: float, read_noise_std: float) -> F64:
    return np.maximum(y, 0.0) / shot_electrons + read_noise_std**2 + 1e-9


def posterior_samples(field: SupportField, count: int, seed: int) -> F64:
    """Measurement-consistent alternative scenes drawn from the Gaussian posterior."""
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(field.cov + 1e-12 * np.eye(field.cov.shape[0]))
    return np.asarray(field.mean[None, :] + rng.standard_normal((count, field.mean.size)) @ chol.T)
