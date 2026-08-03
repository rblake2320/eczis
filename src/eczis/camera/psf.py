# UNFILED-IP
from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def gaussian_kernel(size: int, sigma: float) -> FloatArray:
    if size % 2 == 0 or size < 3:
        raise ValueError("Kernel size must be odd and >= 3")
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx * xx + yy * yy) / np.float32(2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)
