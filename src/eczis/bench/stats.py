# UNFILED-IP
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float], *, confidence: float = 0.95, resamples: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Deterministic percentile-bootstrap CI of the mean (seeded; n reported)."""
    if not values:
        raise ValueError("bootstrap_ci requires at least one value")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    data = np.asarray(values, dtype=np.float64)
    mean = float(data.mean())
    if data.size == 1:
        return {"mean": mean, "lo": mean, "hi": mean, "n": 1.0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(resamples, data.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": mean,
        "lo": float(np.quantile(samples, alpha)),
        "hi": float(np.quantile(samples, 1.0 - alpha)),
        "n": float(data.size),
    }
