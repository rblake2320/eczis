# UNFILED-IP
"""Receive colour + LiDAR depth + camera pose + intrinsics from an iPhone over USB (Record3D app, USB Streaming mode).

    uv run python tools/lidar_stream.py --seconds 8            # records into artifacts/earscan/lidar-<time>/

Writes the same layout the rest of the lab reads, plus what the phone adds:
    frames.npz      grey float32 frames (for the existing tools)
    rgb.npz         colour frames (uint8)
    depth.npz       depth in metres, float32, per frame, from the LiDAR/TrueDepth sensor (INDEPENDENT of any reconstructor)
    poses.txt       TUM: t x y z qx qy qz qw (camera-to-world, from the phone's own tracking)
    lens.json       LensProfile-compatible intrinsics reported by the phone (fx fy cx cy, per stream resolution)
    record.json     device, resolution, frame count, seconds, source = "record3d-usb"

What this changes for the protocol: metric scale and lens come from the phone (statuses CALIBRATED /
SCALED-by-sensor), and the depth map is a measurement that the evidence grader can check the
reconstructor against, instead of comparing the reconstructor with itself. Phone and PC only; nothing
leaves the cable. EXPERIMENTAL; this file is written before it has been run against a phone.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
OUT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "earscan"


def main() -> int:
    from record3d import Record3DStream

    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--device", type=int, default=0, help="index among connected Record3D devices")
    a = ap.parse_args()
    devs = Record3DStream.get_connected_devices()
    if not devs:
        print("no iPhone in Record3D USB streaming mode found (open Record3D -> Settings -> USB Streaming, then press record)")
        return 2
    print("devices:", [(d.product_id, d.udid) for d in devs])
    session = Record3DStream()
    got = threading.Event()
    frames: list[np.ndarray] = []
    rgbs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    poses: list[tuple[float, ...]] = []
    intr: dict[str, float] = {}
    t0 = time.monotonic()

    def on_frame() -> None:
        try:
            depth = session.get_depth_frame()
            rgb = session.get_rgb_frame()
            k = session.get_intrinsic_mat()
            pose = session.get_camera_pose()  # qx qy qz qw tx ty tz
        except Exception as exc:  # noqa: BLE001 - a dropped frame is fine
            print("frame skipped:", exc)
            return
        if not intr:
            intr.update({"fx": float(k.fx), "fy": float(k.fy), "cx": float(k.tx), "cy": float(k.ty)})
        grey = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32) / 255.0
        frames.append(grey)
        rgbs.append(np.asarray(rgb, np.uint8))
        depths.append(np.asarray(depth, np.float32))
        poses.append((time.monotonic() - t0, float(pose.tx), float(pose.ty), float(pose.tz), float(pose.qx), float(pose.qy), float(pose.qz), float(pose.qw)))
        got.set()

    def on_stop() -> None:
        print("stream stopped by the phone")

    session.on_new_frame = on_frame
    session.on_stream_stopped = on_stop
    session.connect(devs[a.device])
    print(f"connected; recording {a.seconds:.0f} s ...", flush=True)
    while time.monotonic() - t0 < a.seconds:
        time.sleep(0.1)
    if not frames:
        print("no frames received")
        return 3
    out = OUT_ROOT / time.strftime("lidar-%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "frames.npz", frames=np.stack(frames))
    np.savez_compressed(out / "rgb.npz", rgb=np.stack(rgbs))
    np.savez_compressed(out / "depth.npz", depth=np.stack(depths))
    (out / "poses.txt").write_text("\n".join(" ".join(f"{v:.6f}" for v in p) for p in poses) + "\n", encoding="utf-8")
    h, w = frames[0].shape
    lens = {"fx": intr["fx"], "fy": intr["fy"], "cx": intr["cx"], "cy": intr["cy"], "dist": [0.0, 0.0, 0.0, 0.0, 0.0], "width": int(w), "height": int(h),
            "rms_px": 0.0, "views": 0, "square_mm": 0.0, "source": "record3d-usb: intrinsics reported by the phone (not solved here)",
            "camera": "iPhone via Record3D", "capture_mode": f"{w}x{h} rgb+depth usb", "model": "pinhole (phone-reported, no distortion terms)"}
    (out / "lens.json").write_text(json.dumps(lens, indent=2), encoding="utf-8")
    (out / "record.json").write_text(json.dumps({"source": "record3d-usb", "frames": len(frames), "seconds": a.seconds, "resolution": [w, h],
                                                 "depth_units": "metres", "depth_median_mm": round(float(np.median(np.stack(depths)) * 1000), 1),
                                                 "pose_path_mm": round(float(sum(np.linalg.norm(np.subtract(poses[i][1:4], poses[i - 1][1:4])) for i in range(1, len(poses))) * 1000), 1)},
                                                indent=2), encoding="utf-8")
    print(f"saved {len(frames)} frames to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
