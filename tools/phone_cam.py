# UNFILED-IP
"""Start the phone camera page and show frames arriving (smoke test for eczis.lab.phone_cam).

    uv run python tools/phone_cam.py            # prints the URL; open it on the phone, allow the camera
    uv run python tools/phone_cam.py --seconds 30

Prints frame rate and size as frames arrive, saves the first frame and any full-res still into
artifacts/phone_cam/. LAN only; self-signed certificate (accept the browser warning once).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    from eczis.lab.coach import CAPTION
    from eczis.lab.phone_cam import CERT_DIR, PhoneCamera

    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--port", type=int, default=8443)
    a = ap.parse_args()
    cam = PhoneCamera(port=a.port, caption_path=CAPTION)
    url = cam.start()
    print(f"Open on the phone (same Wi-Fi): {url}   (accept the certificate warning once)", flush=True)
    t0 = time.monotonic()
    while True:  # keep serving until the phone shows up or the operator stops this
        try:
            cam.wait_for_frame(timeout=30.0)
            break
        except RuntimeError:
            print("waiting for the phone...", flush=True)
    print("first frame after", round(time.monotonic() - t0, 1), "s:", cam.describe(), flush=True)
    from PIL import Image

    rgb = cam.read_display()
    if rgb is not None:
        Image.fromarray(rgb).save(CERT_DIR / "first_frame.png")
    n0, t1 = cam._frames, time.monotonic()
    while time.monotonic() - t0 < a.seconds:
        time.sleep(2.0)
        n1 = time.monotonic()
        fps = (cam._frames - n0) / max(n1 - t1, 1e-6)
        n0, t1 = cam._frames, n1
        print(f"{fps:5.1f} fps  {cam.describe()}", flush=True)
        still = cam._still
        if still is not None and cam._still_t > t0:
            Image.fromarray(still).save(CERT_DIR / "still.png")
            print("full-res still saved:", still.shape, flush=True)
            cam._still = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
