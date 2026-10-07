# UNFILED-IP
"""Calibrate the endoscope lens live from a printed checkerboard.

    uv run python tools/calibrate_camera.py --camera 1 --square 2.0 [--cols 9 --rows 6]

Shows the live view with the board detection drawn on; every time the board is found in a pose
clearly different from the ones already kept, it is accepted automatically (counter on screen).
Tilt and move the board between views. Once >= 8 views are kept, press S to solve and save
artifacts/calib/lens_camera<N>.json, or let it solve automatically at 12. Q quits.
Nothing is uploaded; frames are not stored, only corner positions. EXPERIMENTAL.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "calib"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=2.0, help="measured square size in mm")
    ap.add_argument("--target", type=int, default=12, help="auto-solve after this many views")
    a = ap.parse_args()
    import cv2

    from eczis.lab.calibrate import MIN_VIEWS, Calibrator, summary
    from eczis.lab.sources import OpenCVCamera

    try:
        cam = OpenCVCamera(a.camera, cv2.CAP_MSMF)
        cam.set_mode(640, 480, "YUY2")
        cam.wait_for_frame(4.0)
    except RuntimeError as exc:
        print(f"Camera {a.camera} not streaming ({exc}). Close other camera apps and retry.")
        return 2
    square, print_note = a.square, "no print record (artifacts/calib/print_record.json missing): nominal square size used"
    rec_path = OUT / "print_record.json"
    if rec_path.exists():
        import json

        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        if rec.get("status") != "OK":
            print(f"print record status {rec.get('status')!r}: fix the print before calibrating")
            return 4
        square = a.square * float(rec["print_scale"])
        print_note = f"print_record.json {rec['date']}: bar {rec['bar_measured_h_mm']}x{rec['bar_measured_v_mm']} mm, square {square:.4f} mm"
    print(print_note)
    cal = Calibrator(cols=a.cols, rows=a.rows, square_mm=square)
    win = "ECZIS lens calibration (S solve, Q quit)"
    last_try = 0.0
    try:
        while True:
            try:
                grey = cam.read()
            except RuntimeError:
                time.sleep(0.05)
                continue
            colour = cam.read_display()
            show = cv2.cvtColor(colour, cv2.COLOR_RGB2BGR) if colour is not None else cv2.cvtColor((grey * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            status = f"views kept: {len(cal.detections)}/{a.target}  (need >= {MIN_VIEWS})"
            if time.monotonic() - last_try > 0.4:  # detection is ~100 ms; don't do it every frame
                last_try = time.monotonic()
                corners = cal.detect(grey)
                if corners is not None:
                    cv2.drawChessboardCorners(show, (a.cols, a.rows), corners.astype(np.float32).reshape(-1, 1, 2), True)
                    if cal.add(grey):
                        status += "  ACCEPTED new pose"
                    else:
                        status += "  board seen (move/tilt for a new pose)"
                else:
                    status += "  no board: fill the view with the board, avoid glare"
            cv2.putText(show, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow(win, show)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("s") or len(cal.detections) >= a.target:
                if len(cal.detections) < MIN_VIEWS:
                    print(f"only {len(cal.detections)} views; need {MIN_VIEWS}")
                    continue
                print("solving (with held-out check)...", flush=True)
                prof = cal.solve(camera=f"camera {a.camera} ({cam.describe()})", capture_mode="640x480 YUY2 MSMF", print_record=print_note)
                OUT.mkdir(parents=True, exist_ok=True)
                path = prof.to_json(OUT / f"lens_camera{a.camera}.json")
                s = summary(prof)
                print("lens profile:", s)
                print("saved", path, "| ok" if prof.ok() else "| WARNING: rms too high or too few views; recapture")
                if prof.held_out_stable() is False:
                    print("WARNING: held-out reprojection unstable (a view the solver never saw reprojects badly); recapture")
                if prof.diverse() is False:
                    print("WARNING: views not diverse enough (tilt / position / size); the lens is weakly constrained")
                return 0 if prof.ok() else 3
    finally:
        cv2.destroyAllWindows()
        cam.close()


if __name__ == "__main__":
    raise SystemExit(main())
