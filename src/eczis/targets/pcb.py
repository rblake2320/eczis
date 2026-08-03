# UNFILED-IP
from __future__ import annotations

import numpy as np

from eczis.targets.base import Target


def create_pcb_target(shape: tuple[int, int]) -> Target:
    h, w = shape
    image = np.full(shape, 0.12, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.bool_)
    for y in range(10, h - 10, 17):
        image[y : y + 2, 8 : w - 8] = 0.76
        mask[y : y + 2, 8 : w - 8] = True
    for x in range(12, w - 12, 23):
        image[8 : h - 8, x : x + 2] = 0.56
        mask[8 : h - 8, x : x + 2] = True
    for y in range(20, h - 20, 37):
        for x in range(20, w - 20, 41):
            yy, xx = np.ogrid[:h, :w]
            pad = (xx - x) ** 2 + (yy - y) ** 2 <= 15
            image[pad] = 0.92
    return Target("pcb", image, mask, int(mask.sum()))
