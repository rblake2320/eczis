# UNFILED-IP
"""Probe a USB camera/microscope for ECZIS Microscope Lab suitability. Explicit opt-in only.

    uv run python tools/probe_camera.py --camera 0            # modes + lockable settings
    uv run python tools/probe_camera.py --camera 0 --save DIR # also save 16 raw test frames

Reports which resolutions/pixel formats actually deliver frames (uncompressed YUY2 is preferred:
MJPEG compression breaks the noise model), whether auto exposure / white balance can be disabled,
and frame-to-frame brightness stability. Nothing is saved unless --save is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, required=True)
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args()
    import numpy as np

    from eczis.lab.sources import OpenCVCamera

    cam = OpenCVCamera(args.camera)
    try:
        report: dict[str, Any] = {"device": cam.describe(), "modes": cam.probe_modes(), "settings_lock": cam.lock_settings()}
        frames = cam.capture(16)
        means = [float(f.mean()) for f in frames]
        report["brightness_drift_over_16_frames"] = round(max(means) - min(means), 5)
        report["suitable_uncompressed_modes"] = [m["requested"] for m in report["modes"]
                                                 if m["read_ok"] and m["got_fourcc"].strip("\x00") in ("YUY2", "NV12")]
        if args.save:
            args.save.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                np.save(args.save / f"frame_{i:02d}.npy", f)
        print(json.dumps(report, indent=2, default=str))
    finally:
        cam.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
