# UNFILED-IP
from __future__ import annotations

import hashlib

import numpy as np

from eczis.schemas import BoolArray, FloatArray
from eczis.targets.base import Target

# 5x7 binary glyphs. Small deterministic alphabet gives exact identities without OCR.
_GLYPHS: dict[str, tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
}


def _draw_glyph(canvas: FloatArray, mask: BoolArray, glyph: str, x: int, y: int, stroke: int) -> int:
    count = 0
    for gy, row in enumerate(_GLYPHS[glyph]):
        for gx, bit in enumerate(row):
            if bit == "1":
                ys = slice(y + gy * stroke, y + (gy + 1) * stroke)
                xs = slice(x + gx * stroke, x + (gx + 1) * stroke)
                canvas[ys, xs] = 0.94
                mask[ys, xs] = True
                count += 1
    return count


def create_microglyph_target(
    shape: tuple[int, int], *, seed: int = 20260803, stroke_px: int = 1
) -> Target:
    h, w = shape
    if h < 80 or w < 120:
        raise ValueError("Microglyph target requires at least 80x120 high-resolution pixels")
    rng = np.random.default_rng(seed)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    canvas = (0.08 + 0.06 * xx + 0.03 * yy).astype(np.float32)
    # Fine carrier texture makes registration possible without replacing exact glyph truth.
    canvas += (0.012 * np.sin(2 * np.pi * (11 * xx + 7 * yy))).astype(np.float32)
    canvas += rng.normal(0.0, 0.003, shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.bool_)

    letters = tuple(_GLYPHS)
    glyph_h = 7 * stroke_px
    glyph_w = 5 * stroke_px
    gap_x = max(4, 3 * stroke_px)
    gap_y = max(5, 4 * stroke_px)
    margin = 8
    x = margin
    y = margin
    feature_count = 0
    index = 0
    while y + glyph_h + margin < h:
        while x + glyph_w + margin < w:
            glyph = letters[(index + seed) % len(letters)]
            feature_count += _draw_glyph(canvas, mask, glyph, x, y, stroke_px)
            x += glyph_w + gap_x
            index += 1
        x = margin + ((y // max(1, gap_y)) % 3)
        y += glyph_h + gap_y

    # Fiducials and one high-contrast diagonal for shift registration.
    canvas[4:10, 4:10] = 1.0
    canvas[-10:-4, -10:-4] = 1.0
    for i in range(min(h, w)):
        if i % 9 == 0:
            canvas[i : i + 1, max(0, w - i - 2) : max(0, w - i - 1)] = 0.75
    canvas = np.clip(canvas, 0.0, 1.0).astype(np.float32)
    digest = hashlib.sha256(canvas.tobytes()).hexdigest()[:16]
    return Target(f"microglyph-{digest}", canvas, mask, feature_count)
