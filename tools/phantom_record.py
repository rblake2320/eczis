# UNFILED-IP
"""Record what was scanned: tube measurements and the role of a sweep in the protocol.

    uv run python tools/phantom_record.py artifacts/earscan/<time> --role straight --diameter-mm 8.0 \\
        --material "clear PVC, matte inside" --length-mm 60 --insertion-mm 5.0 --led 3 --sweeps 1 --duration-s 6 --centred yes

Roles: straight | twist | repeat-twist | low-motion-control | high-glare-control | wrong-calibration-control
Writes <folder>/phantom_record.json; the benchmark report reads the role and the known values from it,
so the operator never retypes the diameter and the controls are identified by record, not by memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROLES = ("straight", "twist", "repeat-twist", "low-motion-control", "high-glare-control", "wrong-calibration-control")


def record(folder: Path, **fields: object) -> dict[str, object]:
    rec: dict[str, object] = {"folder": folder.name, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    frames = folder / "frames.npz"
    if frames.exists():
        rec["frames_sha256"] = hashlib.sha256(frames.read_bytes()).hexdigest()
        rec["frames_bytes"] = frames.stat().st_size
    (folder / "phantom_record.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--role", choices=ROLES, required=True)
    ap.add_argument("--diameter-mm", type=float, required=True, help="inner diameter measured physically (calipers)")
    ap.add_argument("--material", required=True)
    ap.add_argument("--length-mm", type=float, required=True, help="visible length of the tube")
    ap.add_argument("--insertion-mm", type=float, required=True, help="probe translation between the start and end marks")
    ap.add_argument("--led", required=True, help="LED wheel setting as read")
    ap.add_argument("--sweeps", type=int, default=1)
    ap.add_argument("--duration-s", type=float, required=True)
    ap.add_argument("--centred", choices=["yes", "no", "unknown"], required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if not a.folder.exists():
        print(f"{a.folder} does not exist", file=sys.stderr)
        return 2
    rec = record(a.folder, role=a.role, diameter_mm=a.diameter_mm, material=a.material, length_mm=a.length_mm,
                 insertion_mm=a.insertion_mm, led=a.led, sweeps=a.sweeps, duration_s=a.duration_s, centred=a.centred, note=a.note)
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
