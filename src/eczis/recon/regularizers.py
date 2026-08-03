# UNFILED-IP
from __future__ import annotations

import numpy as np

from eczis.schemas import FloatArray


def laplacian(image: FloatArray) -> FloatArray:
    return (
        4.0 * image
        - np.roll(image, 1, axis=0)
        - np.roll(image, -1, axis=0)
        - np.roll(image, 1, axis=1)
        - np.roll(image, -1, axis=1)
    ).astype(np.float32)
