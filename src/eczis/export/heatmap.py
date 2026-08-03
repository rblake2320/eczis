# UNFILED-IP
from __future__ import annotations

import numpy as np
from PIL import Image

from eczis.schemas import UInt8Array


def provenance_rgba(provenance: UInt8Array, alpha: int = 150) -> Image.Image:
    palette = np.array(
        [
            [20, 20, 20, alpha],      # U
            [245, 158, 11, alpha],    # I
            [37, 99, 235, alpha],     # R
            [34, 197, 94, alpha],     # M
        ],
        dtype=np.uint8,
    )
    rgba = palette[np.clip(provenance, 0, 3)]
    return Image.fromarray(rgba, mode="RGBA")
