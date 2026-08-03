# UNFILED-IP
from __future__ import annotations

import numpy as np

from eczis.targets.base import Target


def create_coin_target(shape: tuple[int, int]) -> Target:
    h, w = shape
    y, x = np.mgrid[:h, :w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.hypot((x - cx) / (0.43 * w), (y - cy) / (0.43 * h))
    theta = np.arctan2(y - cy, x - cx)
    relief = 0.5 + 0.18 * np.cos(18 * theta) * np.exp(-2.2 * r)
    rings = 0.10 * np.sin(55 * r)
    image = 0.08 + (relief + rings) * (r <= 1.0)
    scratch = np.abs((y - cy) - 0.34 * (x - cx)) < 0.8
    image[scratch & (r < 0.9)] = 0.95
    mask = scratch & (r < 0.9)
    return Target("coin", np.clip(image, 0, 1).astype(np.float32), mask.astype(np.bool_), int(mask.sum()))
