# UNFILED-IP
"""Explicit colour-filter-array geometry.

The v0.1R1 demonstration is monochrome; this module freezes CFA sample
locations so a colour capture can never rely on a hidden global default.
Patterns are named by the 2x2 tile read row-major from the top-left pixel.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

CfaPattern = Literal["RGGB", "BGGR", "GRBG", "GBRG", "MONO"]
CFA_PATTERNS: tuple[str, ...] = ("RGGB", "BGGR", "GRBG", "GBRG", "MONO")
_CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2}


def validate_pattern(pattern: str) -> CfaPattern:
    if pattern not in CFA_PATTERNS:
        raise ValueError(f"Unknown CFA pattern {pattern!r}; expected one of {CFA_PATTERNS}")
    return pattern  # type: ignore[return-value]


def channel_masks(shape: tuple[int, int], pattern: str) -> npt.NDArray[np.bool_]:
    """Return a ``(3, h, w)`` boolean stack: which pixels sample R, G, B."""
    validate_pattern(pattern)
    h, w = shape
    masks = np.zeros((3, h, w), dtype=np.bool_)
    if pattern == "MONO":
        masks[:] = True
        return masks
    for index, letter in enumerate(pattern):
        oy, ox = divmod(index, 2)
        masks[_CHANNEL_INDEX[letter], oy::2, ox::2] = True
    return masks


def mosaic(rgb: npt.NDArray[np.float32], pattern: str) -> npt.NDArray[np.float32]:
    """Sample an ``(h, w, 3)`` linear scene through the CFA into an ``(h, w)`` RAW plane."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("mosaic expects an (h, w, 3) array")
    masks = channel_masks((rgb.shape[0], rgb.shape[1]), pattern)
    if pattern == "MONO":
        return rgb.mean(axis=2).astype(np.float32)
    raw = np.zeros(rgb.shape[:2], dtype=np.float32)
    for channel in range(3):
        raw[masks[channel]] = rgb[..., channel][masks[channel]]
    return raw
