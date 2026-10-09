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


def board_geometry(corners: np.ndarray, cols: int, rows: int) -> tuple[float, float, float]:
    """(mean corner spacing px, apparent width px, apparent height px) of a detected board."""
    grid = corners.reshape(rows, cols, 2)
    sx = float(np.mean(np.linalg.norm(np.diff(grid, axis=1), axis=2)))
    sy = float(np.mean(np.linalg.norm(np.diff(grid, axis=0), axis=2)))
    return 0.5 * (sx + sy), sx * (cols - 1), sy * (rows - 1)


def live_readout(corners: np.ndarray, cols: int, rows: int, square_mm: float, width_px: int, fx: float | None = None) -> str:
    """Approximate board distance and tilt from one detection.

    Distance uses the calibrated fx when known, else the 0.8 x width guess (marked '~'): it is a
    positioning aid for the operator, never a measurement that enters the profile.
    """
    spacing, w, h = board_geometry(corners, cols, rows)
    f = fx if fx is not None else 0.8 * width_px
    dist = f * square_mm / max(spacing, 1e-6)
    sx = w / (cols - 1)
    sy = h / (rows - 1)
    fore = min(sx, sy) / max(sx, sy, 1e-6)  # foreshortening of the shorter axis
    tilt = float(np.degrees(np.arccos(np.clip(fore, 0.0, 1.0))))
    mark = "" if fx is not None else "~"
    return f"distance {mark}{dist:.0f} mm   tilt ~{tilt:.0f} deg   board {w:.0f}x{h:.0f} px"


MAX_FLAT_VIEWS = 4  # after this many views under 15 deg tilt, flat views are refused
MAX_CENTRED_VIEWS = 4  # after this many views in the centre region, centred views are refused
TILT_WANTED_DEG = 15.0


def view_tilt(corners: np.ndarray, cols: int, rows: int) -> float:
    _, w, h = board_geometry(corners, cols, rows)
    sx, sy = w / (cols - 1), h / (rows - 1)
    return float(np.degrees(np.arccos(np.clip(min(sx, sy) / max(sx, sy, 1e-6), 0, 1))))


def view_centred(corners: np.ndarray, shape: tuple[int, ...]) -> bool:
    cx, cy = corners.mean(axis=0) / [shape[1], shape[0]]
    return bool(0.35 <= cx <= 0.65 and 0.35 <= cy <= 0.65)


def view_wanted(corners: np.ndarray, kept: list[np.ndarray], cols: int, rows: int, shape: tuple[int, ...]) -> tuple[bool, str]:
    """Refuse the kinds of view the set already has enough of, so it cannot fill with flat centred boards."""
    flat = sum(view_tilt(d, cols, rows) < TILT_WANTED_DEG for d in kept)
    centred = sum(view_centred(d, shape) for d in kept)
    if view_tilt(corners, cols, rows) < TILT_WANTED_DEG and flat >= MAX_FLAT_VIEWS:
        return False, f"enough flat views ({flat}): tilt the board to {TILT_WANTED_DEG:.0f} degrees or more"
    if view_centred(corners, shape) and centred >= MAX_CENTRED_VIEWS:
        return False, f"enough centred views ({centred}): move the board to an edge or corner"
    return True, ""


def coverage_readout(detections: list[np.ndarray], cols: int, rows: int, shape: tuple[int, ...]) -> str:
    """Which image regions / tilts / sizes the kept views already cover, and what is still needed."""
    if not detections:
        return ""
    h_img, w_img = shape[0], shape[1]
    regions = {"C": False, "L": False, "R": False, "T": False, "B": False}
    tilts, sizes = [], []
    for d in detections:
        cx, cy = d.mean(axis=0) / [w_img, h_img]
        regions["L" if cx < 0.35 else "R" if cx > 0.65 else "C"] = True
        if cy < 0.35:
            regions["T"] = True
        if cy > 0.65:
            regions["B"] = True
        spacing, w, hh = board_geometry(d, cols, rows)
        sx, sy = w / (cols - 1), hh / (rows - 1)
        tilts.append(float(np.degrees(np.arccos(np.clip(min(sx, sy) / max(sx, sy, 1e-6), 0, 1)))))
        sizes.append(spacing)
    names = {"C": "centre", "L": "left", "R": "right", "T": "top", "B": "bottom"}
    have = ", ".join(names[k] for k, v in regions.items() if v)
    need = ", ".join(names[k] for k, v in regions.items() if not v)
    size_ratio = max(sizes) / max(min(sizes), 1e-6)
    todo = []
    if need:
        todo.append(f"move board to the {need}")
    if max(tilts) < 15:
        todo.append(f"tilt more (max {max(tilts):.0f}, need 15+)")
    if size_ratio < 1.25:
        todo.append(f"vary distance (size range {size_ratio:.2f}x, need 1.25x)")
    return f"covered {have or '-'} | tilt max {max(tilts):.0f} deg | size range {size_ratio:.2f}x | " + ("; ".join(todo) if todo else "diverse enough")


def wrap(text: str, width_px: int, scale: float, thick: int = 2) -> list[str]:
    import cv2

    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] > width_px and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def draw_zones(img: np.ndarray, coverage: str) -> None:
    """Centre crosshair plus the five zones (centre, left, right, top, bottom); zones still needed are outlined in orange."""
    import cv2

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 1)
    cv2.line(img, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 1)
    cv2.circle(img, (cx, cy), 6, (255, 255, 255), 1)
    need = coverage.split("move board to the ")[1].split(";")[0].split("|")[0] if "move board to the " in coverage else ""
    boxes = {"centre": (int(0.35 * w), int(0.35 * h), int(0.65 * w), int(0.65 * h)),
             "left": (4, int(0.35 * h), int(0.35 * w), int(0.65 * h)), "right": (int(0.65 * w), int(0.35 * h), w - 4, int(0.65 * h)),
             "top": (int(0.35 * w), 4, int(0.65 * w), int(0.35 * h)), "bottom": (int(0.35 * w), int(0.65 * h), int(0.65 * w), h - 4)}
    for name, (x0, y0, x1, y1) in boxes.items():
        wanted = name in need
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 140, 255) if wanted else (90, 90, 90), 2 if wanted else 1)
        if wanted:
            cv2.putText(img, name.upper(), (x0 + 8, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)


def compose(frame: np.ndarray, banner: list[str], caption: str, scale: int = 2) -> np.ndarray:
    """Banner bar (status / live / coverage, wrapped) + picture scaled up + caption band at the bottom."""
    import cv2

    img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    draw_zones(img, banner[2] if len(banner) > 2 else "")
    w = img.shape[1]
    fs, lh = 0.7, 30
    colours = [(0, 255, 0), (0, 255, 255), (255, 200, 0)]
    banner_lines: list[tuple[str, tuple[int, int, int]]] = []
    for text, col in zip(banner, colours, strict=True):
        banner_lines += [(ln, col) for ln in wrap(text, w - 20, fs)] if text else []
    top = np.zeros((lh * max(len(banner_lines), 1) + 16, w, 3), np.uint8)
    for i, (ln, col) in enumerate(banner_lines):
        cv2.putText(top, ln, (10, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 2)
    cap_lines = wrap(caption, w - 40, 1.0) if caption else []
    bottom = np.zeros((44 * max(len(cap_lines), 1) + 20, w, 3), np.uint8)
    for i, ln in enumerate(cap_lines):
        tw = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
        cv2.putText(bottom, ln, ((w - tw) // 2, 44 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return np.vstack([top, img, bottom])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=2.0, help="measured square size in mm")
    ap.add_argument("--target", type=int, default=12, help="auto-solve after this many views")
    ap.add_argument("--quiet", action="store_true", help="no spoken guidance / captions")
    ap.add_argument("--scale", type=int, default=2, help="window magnification")
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
    live = ""  # distance / tilt of the board seen right now
    coverage = ""  # what the kept views already cover and what is still needed
    from eczis.lab.coach import Coach

    coach = Coach(voice=not a.quiet)
    seen, needs = "", ""
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
                    live = live_readout(corners, a.cols, a.rows, square, grey.shape[1])
                    seen = "board"
                    kind_ok, why = view_wanted(corners, cal.detections, a.cols, a.rows, grey.shape)
                    if not kind_ok:
                        status += f"  {why}"
                        needs = why
                    elif cal.add(grey):
                        status += "  ACCEPTED new pose"
                    else:
                        status += "  board seen (move/tilt for a new pose)"
                    coverage = coverage_readout(cal.detections, a.cols, a.rows, grey.shape)
                    needs = coverage.split("| ")[-1]
                else:
                    status += "  no board: fill the view with the board, avoid glare"
                    live = ""
                    seen = "none"
            guidance = "" if seen != "board" else ("hold still" if "ACCEPTED" in status else (needs if needs and needs != "diverse enough" else ""))
            words = coach.tick({"window": "calibrate", "views": len(cal.detections), "target": a.target, "sees": seen,
                                "live": live, "coverage": coverage}, guidance)
            cv2.imshow(win, compose(show, [status, live, coverage], words, a.scale))
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                return 0
            diverse_now = coverage.endswith("diverse enough")
            if key == ord("s") or (len(cal.detections) >= a.target and diverse_now):
                if len(cal.detections) < MIN_VIEWS:
                    print(f"only {len(cal.detections)} views; need {MIN_VIEWS}")
                    continue
                print("solving (with held-out check)...", flush=True)
                prof = cal.solve(camera=f"camera {a.camera} ({cam.describe()})", capture_mode="640x480 YUY2 MSMF", print_record=print_note)
                OUT.mkdir(parents=True, exist_ok=True)
                path = prof.to_json(OUT / f"lens_camera{a.camera}.json")
                s = summary(prof)
                print("lens profile:", s)
                print("saved", path, "| usable" if prof.usable() else "| REJECTED: not usable, recapture (see warnings)")
                if not prof.plausible():
                    print(f"WARNING: field of view {prof.hfov_deg():.1f} deg is implausible: views too front-facing/centred; tilt 20-30 deg")
                if not prof.ok():
                    print("WARNING: rms too high or too few views")
                if prof.held_out_stable() is False:
                    print("WARNING: held-out reprojection unstable (a view the solver never saw reprojects badly); recapture")
                if prof.diverse() is False:
                    print("WARNING: views not diverse enough (tilt / position / size); the lens is weakly constrained")
                if not prof.usable():
                    path.rename(path.with_name(path.stem + f"_rejected_{time.strftime('%H%M%S')}.json"))
                return 0 if prof.usable() else 3
    finally:
        cv2.destroyAllWindows()
        cam.close()


if __name__ == "__main__":
    raise SystemExit(main())
