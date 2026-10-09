# UNFILED-IP
"""Make a true-size checkerboard page for lens calibration.

    uv run python tools/checkerboard_print.py                 # 9x6 inner corners, 2.0 mm squares, A4/Letter PNG+PDF
    uv run python tools/checkerboard_print.py --square 1.5 --cols 9 --rows 6

Print at 100% (no "fit to page"), then measure one square with a ruler and pass the measured
size to the calibrator if it differs. Small squares suit the endoscope's close working distance
(~1-3 cm); glue the page to something flat. Output: artifacts/calib/checkerboard_<cols>x<rows>_<mm>mm.{png,pdf}
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "calib"


def make(cols: int, rows: int, square_mm: float, dpi: int = 600, page: str = "letter") -> tuple[Path, Path]:
    page_in = {"letter": (8.5, 11.0), "a4": (8.27, 11.69)}[page]
    w, h = int(page_in[0] * dpi), int(page_in[1] * dpi)
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    px = square_mm / 25.4 * dpi  # pixels per square
    nx, ny = cols + 1, rows + 1  # squares = inner corners + 1
    bw, bh = nx * px, ny * px
    x0, y0 = (w - bw) / 2, (h - bh) / 2
    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 == 0:
                d.rectangle((x0 + i * px, y0 + j * px, x0 + (i + 1) * px - 1, y0 + (j + 1) * px - 1), fill=0)
    # a 20 mm scale bar for verifying print size with a ruler
    bar = 20 / 25.4 * dpi
    bx, by = x0, y0 + bh + 0.4 * dpi
    d.rectangle((bx, by, bx + bar, by + 0.05 * dpi), fill=0)
    d.text((bx, by + 0.08 * dpi), f"20 mm scale bar  |  {cols}x{rows} inner corners, {square_mm} mm squares  |  print at 100%", fill=0)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"checkerboard_{cols}x{rows}_{square_mm:g}mm"
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    img.save(png, dpi=(dpi, dpi))
    img.convert("RGB").save(pdf, "PDF", resolution=dpi)
    return png, pdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=2.0, help="square size in mm")
    ap.add_argument("--page", choices=["letter", "a4"], default="letter")
    a = ap.parse_args()
    png, pdf = make(a.cols, a.rows, a.square, page=a.page)
    print(f"wrote {png}\nwrote {pdf}\nPrint at 100% scale; check the 20 mm bar with a ruler.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
