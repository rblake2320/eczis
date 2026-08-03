# UNFILED-IP
from __future__ import annotations

import numpy as np

from eczis.targets.base import Target


def create_siemens_target(shape: tuple[int, int], spokes: int = 48) -> Target:
    h, w = shape
    y, x = np.mgrid[:h, :w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    angle = np.arctan2(y - cy, x - cx)
    radius = np.hypot(x - cx, y - cy)
    star = 0.5 + 0.42 * np.sign(np.sin(spokes * angle / 2.0))
    envelope = np.clip(1.0 - radius / max(h, w), 0.25, 1.0)
    image = (0.08 + star * envelope).astype(np.float32)
    mask = radius < min(h, w) * 0.45
    return Target("siemens", np.clip(image, 0, 1), mask.astype(np.bool_), int(mask.sum()))
