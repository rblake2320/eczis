# UNFILED-IP
from __future__ import annotations

import numpy as np

from eczis.targets.base import Target


def create_iris_target(shape: tuple[int, int], seed: int = 77) -> Target:
    h, w = shape
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:h, :w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    radius = np.hypot(x - cx, y - cy)
    angle = np.arctan2(y - cy, x - cx)
    outer = min(h, w) * 0.43
    pupil = min(h, w) * 0.13
    fibers = 0.22 * np.sin(85 * angle + 0.16 * radius) + 0.08 * np.sin(0.9 * radius)
    noise = rng.normal(0, 0.025, shape)
    iris = 0.42 + fibers + noise
    image = np.full(shape, 0.10, dtype=np.float32)
    ring = (radius <= outer) & (radius >= pupil)
    image[ring] = np.clip(iris[ring], 0.12, 0.9)
    image[radius < pupil] = 0.01
    mask = ring & (np.abs(fibers) > 0.18)
    return Target("iris", image, mask.astype(np.bool_), int(mask.sum()))
