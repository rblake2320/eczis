# UNFILED-IP
"""Record the printed-target verification (the 20 mm scale bar measured with a ruler, both axes).

    uv run python tools/print_record.py --printer "Brother HL-L2350DW" --paper "letter 20lb" \\
        --viewer "Adobe Reader" --scaling "actual size, 100%" --h-mm 20.0 --v-mm 20.0

Writes artifacts/calib/print_record.json. The calibrator uses the measured bar to correct the
square size (nominal 2.0 mm x measured/20). If the two axes differ by more than 1% the record is
written with status MISMATCH and exit 2: fix the print, do not compensate in software.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "calib" / "print_record.json"
BAR_MM = 20.0
MAX_AXIS_MISMATCH = 0.01


def record(printer: str, paper: str, viewer: str, scaling: str, h_mm: float, v_mm: float, nominal_square_mm: float = 2.0,
           path: Path = OUT) -> dict[str, object]:
    mismatch = abs(h_mm - v_mm) / BAR_MM
    scale = (h_mm + v_mm) / 2 / BAR_MM
    rec = {"printer": printer, "paper": paper, "viewer": viewer, "scaling": scaling, "date": time.strftime("%Y-%m-%d"),
           "bar_nominal_mm": BAR_MM, "bar_measured_h_mm": h_mm, "bar_measured_v_mm": v_mm,
           "axis_mismatch_frac": round(mismatch, 4), "print_scale": round(scale, 5),
           "nominal_square_mm": nominal_square_mm, "effective_square_mm": round(nominal_square_mm * scale, 5),
           "status": "MISMATCH: axes differ, fix the print" if mismatch > MAX_AXIS_MISMATCH else "OK"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("printer", "paper", "viewer", "scaling"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--h-mm", type=float, required=True, help="measured horizontal length of the 20 mm bar")
    ap.add_argument("--v-mm", type=float, required=True, help="measured vertical length of the 20 mm bar (rotate the ruler)")
    ap.add_argument("--square", type=float, default=2.0, help="nominal square size printed")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rec = record(a.printer, a.paper, a.viewer, a.scaling, a.h_mm, a.v_mm, a.square, a.out)
    print(json.dumps(rec, indent=2))
    if rec["status"] != "OK":
        print(str(rec["status"]), file=sys.stderr)
        return 2
    print(f"effective square {rec['effective_square_mm']} mm -> calibrate_camera.py picks this up automatically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
