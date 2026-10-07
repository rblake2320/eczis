# UNFILED-IP
"""Record a handheld canal sweep (e.g. ear) and reconstruct an evidence-graded 3D point cloud.

    uv run python tools/ear_scan.py                  # camera 1 (endoscope), Media Foundation
    uv run python tools/ear_scan.py --simulate       # synthetic canal, no camera (dry run)

Live preview: SPACE = record a sweep, Q / Esc = quit. During recording move slowly in and out
with a slight twist. Personal media: frames and results go ONLY to artifacts/earscan/<time>/
(gitignored); nothing is uploaded or committed. Output is UNCALIBRATED (lens guessed) and has
NO SCALE (arbitrary units) until calibration and a known insertion distance exist. Measurement
record only, not a diagnosis. EXPERIMENTAL.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "earscan"


def analyse(frames: list[np.ndarray], out: Path, simulated: object = None) -> dict[str, object]:
    from eczis.lab.scan3d import Intrinsics, export_ply, reconstruct, summarize

    h, w = frames[0].shape
    if simulated is not None:
        k = simulated.k  # type: ignore[attr-defined]
        base = float(np.linalg.norm(simulated.centres[6] - simulated.centres[0]))  # type: ignore[attr-defined]
        notes = ["synthetic canal: true lens and scale"]
    else:
        k = Intrinsics(0.8 * w, 0.8 * w, w / 2.0, h / 2.0, w, h)
        base = 1.0
        notes = ["UNCALIBRATED lens (focal length guessed as 0.8 x width)",
                 "NO SCALE: distances in arbitrary units until a known insertion distance is supplied",
                 "measurement record only, not a diagnosis"]
    t0 = time.perf_counter()
    res = reconstruct(frames, k, base, init_gap=6 if simulated is not None else None)
    summary = summarize(res)
    summary["notes"] = list(summary["notes"]) + notes
    summary["seconds"] = round(time.perf_counter() - t0, 2)
    summary["frames_recorded"] = len(frames)
    export_ply(res.points, str(out / "scan.ply"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def quality(frame: np.ndarray) -> tuple[str, str]:
    from scipy.ndimage import laplace

    clipped = float(np.mean((frame <= 0.002) | (frame >= 0.998)))
    focus = float(np.var(laplace(frame.astype(np.float32))))
    return ("GOOD" if clipped < 0.05 else f"CLIPPING {clipped:.0%}"), ("GOOD" if focus > 2e-4 else "SOFT")


def run_camera(index: int, seconds: float, stride: int, auto: float | None = None) -> int:
    import cv2

    from eczis.lab.sources import OpenCVCamera

    try:
        cam = OpenCVCamera(index, cv2.CAP_MSMF)
        cam.set_mode(640, 480, "YUY2")
        cam.wait_for_frame(4.0)
    except RuntimeError as exc:
        print(f"Camera {index} not streaming ({exc}). Close the Lab window / Windows Camera app and retry.")
        return 2
    print("Preview open. SPACE = record sweep, Q/Esc = quit.", flush=True)
    win = "ECZIS ear scan (SPACE record, Q quit)"
    t_open = time.monotonic()
    try:
        while True:
            try:
                grey = cam.read()
            except RuntimeError:
                time.sleep(0.05)
                continue
            colour = cam.read_display()
            show = cv2.cvtColor(colour, cv2.COLOR_RGB2BGR) if colour is not None else (grey * 255).astype(np.uint8)
            exp, foc = quality(grey)
            cv2.putText(show, f"Exposure {exp}  Focus {foc}  |  SPACE record  Q quit", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            remaining = None if auto is None else auto - (time.monotonic() - t_open)
            if remaining is not None and remaining > 0:
                cv2.putText(show, f"RECORDING IN {remaining:0.0f}", (150, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
            cv2.imshow(win, show)
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord(" ") or (remaining is not None and remaining <= 0):
                n = int(seconds * 30)
                print(f"Recording {seconds:.0f} s: move slowly in and out with a slight twist...", flush=True)
                rec = show.copy()
                cv2.putText(rec, "RECORDING - move slowly", (90, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow(win, rec)
                cv2.waitKey(1)
                out = OUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
                out.mkdir(parents=True, exist_ok=True)
                frames, snaps = [], 0
                for i in range(n):  # grey frames for 3D, full-colour stills ~every 0.5 s for close inspection
                    frames.extend(cam.capture(1))
                    if i % 15 == 0:
                        colour_now = cam.read_display()
                        if colour_now is not None:
                            cv2.imwrite(str(out / f"snapshot_{snaps:02d}.png"), cv2.cvtColor(colour_now, cv2.COLOR_RGB2BGR))
                            snaps += 1
                frames = frames[::stride]
                np.savez_compressed(out / "frames.npz", frames=np.stack(frames).astype(np.float32))
                print(f"Recorded {len(frames)} frames -> reconstructing...")
                try:
                    summary = analyse([np.asarray(f, np.float64) for f in frames], out)
                except Exception as exc:  # noqa: BLE001 - report OpenCV failures plainly
                    print(f"Reconstruction failed: {exc}. Try a slower sweep with the canal wall in view.")
                    continue
                print(json.dumps(summary, indent=2), flush=True)
                print(f"Saved to {out}: scan.ply (Blender/MeshLab) + {snaps} colour snapshots.", flush=True)
                if auto is not None:
                    return 0
    finally:
        cv2.destroyAllWindows()
        cam.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--simulate", action="store_true", help="dry run on the synthetic canal")
    ap.add_argument("--auto", type=float, default=None, help="start recording automatically after N seconds")
    args = ap.parse_args()
    if args.simulate:
        from eczis.lab.scan3d import simulate_sweep

        sim = simulate_sweep(frames=40, depth_mm=4.0)
        out = OUT_ROOT / ("sim-" + time.strftime("%Y%m%d-%H%M%S"))
        out.mkdir(parents=True, exist_ok=True)
        print(json.dumps(analyse(sim.frames, out, sim), indent=2))
        print(f"Saved to {out}")
        return 0
    return run_camera(args.camera, args.seconds, args.stride, args.auto)


if __name__ == "__main__":
    raise SystemExit(main())
