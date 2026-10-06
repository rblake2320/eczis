# UNFILED-IP
"""EXPERIMENTAL: answer certificates over measurement-consistent scenes.

A question is a linear functional g (weights over HR pixels) and a decision threshold. Over the
Gaussian posterior's kappa-sigma ellipsoid {x : (x - mu)^T C^-1 (x - mu) <= kappa^2} the extreme
answers are exactly g.mu +/- kappa * sqrt(g^T C g) (constrained optimisation in closed form).
Two further mechanisms cross-check the interval: posterior sampling and a null-direction push
that maximises g while leaving the predicted observations unchanged.

Decisions: SUPPORTED_ABOVE / SUPPORTED_BELOW if the whole interval is on one side of the
threshold, otherwise UNDETERMINED (with RECAPTURE suggested when a candidate measurement would
narrow it). The question interface is deliberately generic so further certificate types
(crack width, trace continuity, digit identity, counts, change) can be added as functionals.
Not used by production classification.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from eczis.experimental.support import F64, SupportField, posterior_samples


class Decision(StrEnum):
    SUPPORTED_ABOVE = "supported_above"
    SUPPORTED_BELOW = "supported_below"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True, slots=True)
class Question:
    name: str
    weights: F64  # flat HR weights defining g(x) = weights . x
    threshold: float
    above_means: str = "present"
    below_means: str = "absent"


@dataclass(frozen=True, slots=True)
class Certificate:
    question: str
    estimate: float
    lower: float
    upper: float
    decision: Decision
    answer: str
    sample_range: tuple[float, float]
    null_push_range: tuple[float, float]
    notes: list[str] = field(default_factory=list)


def certify(
    sf: SupportField, q: Question, *, kappa: float = 3.0, samples: int = 200, seed: int = 0
) -> Certificate:
    g = q.weights.ravel()
    est = float(g @ sf.mean)
    sd = float(np.sqrt(max(float(g @ sf.cov @ g), 0.0)))
    lo, hi = est - kappa * sd, est + kappa * sd
    if lo > q.threshold:
        decision, answer = Decision.SUPPORTED_ABOVE, q.above_means
    elif hi < q.threshold:
        decision, answer = Decision.SUPPORTED_BELOW, q.below_means
    else:
        decision, answer = Decision.UNDETERMINED, "undetermined"
    vals = posterior_samples(sf, samples, seed) @ g
    # Null-direction push: x = mu + t * C g / sqrt(g^T C g) moves along the direction the
    # measurements constrain least for this question; at t = +/- kappa it reaches the interval ends.
    direction = sf.cov @ g / max(sd, 1e-12)
    push = [float(g @ (sf.mean + t * direction)) for t in (-kappa, kappa)]
    return Certificate(q.name, est, lo, hi, decision, answer,
                       (float(vals.min()), float(vals.max())), (min(push), max(push)))


def best_next_measurement(
    current_precision: F64, candidates: Sequence[tuple[str, F64, F64]], g: F64
) -> list[tuple[str, float]]:
    """Rank candidate measurements (name, A_rows, noise_var) by resulting answer std for g."""
    ranked = []
    for name, rows, var in candidates:
        prec = current_precision + (rows.T / np.maximum(var, 1e-12)) @ rows
        cov = np.linalg.inv(prec)
        ranked.append((name, float(np.sqrt(max(float(g @ cov @ g), 0.0)))))
    return sorted(ranked, key=lambda item: item[1])
