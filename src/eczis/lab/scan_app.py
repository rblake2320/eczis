# UNFILED-IP
"""ECZIS Ear / Canal Scan window (PySide6). EXPERIMENTAL.

    uv run python -m eczis.lab.scan_app                # endoscope = camera 1 (Media Foundation)
    uv run python -m eczis.lab.scan_app --camera 0
    uv run python -m eczis.lab.scan_app --simulate     # synthetic canal, no camera

Buttons: Record sweep (countdown, then records), Snapshot (full-resolution colour still),
Ultra-zoom (click the live view to magnify that spot), Open 3D / Open folder. Reconstruction
runs on a worker thread so the window never freezes. Results go to artifacts/earscan/<time>/
(gitignored, local only). Output is UNCALIBRATED / NO SCALE until calibration exists.
Measurement record only, not a diagnosis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # the window reports camera problems itself; no WARN flood
import time
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

OUT_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "earscan"
VIEW_W, VIEW_H = 640, 480
ZOOM_SIDE = 320


def to_pixmap(img: np.ndarray, w: int, h: int, smooth: bool = True) -> QPixmap:
    a = np.ascontiguousarray(img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8))
    fmt = QImage.Format.Format_Grayscale8 if a.ndim == 2 else QImage.Format.Format_RGB888
    q = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], fmt).copy()
    mode = Qt.TransformationMode.SmoothTransformation if smooth else Qt.TransformationMode.FastTransformation
    return QPixmap.fromImage(q).scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, mode)


def guidance(grey: np.ndarray) -> tuple[str, str, str]:
    from scipy.ndimage import laplace

    clipped = float(np.mean(grey >= 0.98))
    dark = float(np.mean(grey <= 0.03))
    focus = float(np.var(laplace(grey.astype(np.float32))))
    exp = "GOOD" if clipped < 0.05 and dark < 0.5 else ("GLARE" if clipped >= 0.05 else "TOO DARK")
    foc = "SHARP" if focus > 2e-4 else "SOFT"
    tip = {"GLARE": "Turn the LED wheel down", "TOO DARK": "Turn the LED wheel up"}.get(exp, "")
    if not tip and foc == "SOFT":
        tip = "Back off slightly until the image is sharp"
    return exp, foc, tip or "Ready: Record, then move slowly in and out with a slight twist"


class SyntheticSource:
    """Plays a synthetic canal sweep in a loop, with the camera API the window expects."""

    def __init__(self) -> None:
        from eczis.lab.scan3d import simulate_sweep

        self.sim = simulate_sweep(frames=60)
        self.i = 0

    def read(self) -> np.ndarray:
        f = self.sim.frames[self.i % len(self.sim.frames)]
        self.i += 1
        return np.asarray(f, np.float32)

    def read_display(self) -> np.ndarray:
        g = (np.clip(self.sim.frames[self.i % len(self.sim.frames)], 0, 1) * 255).astype(np.uint8)
        return np.stack([g, (g * 0.85).astype(np.uint8), (g * 0.75).astype(np.uint8)], axis=-1)

    def capture(self, count: int) -> list[np.ndarray]:
        out = list(self.sim.frames[:count]) if count <= len(self.sim.frames) else list(self.sim.frames)
        self.i = 0
        return [np.asarray(f, np.float32) for f in out]

    def close(self) -> None:
        pass


class CaptureWorker(QObject):
    """Grabs the sweep's frames off the UI thread so the live view keeps moving while recording."""

    done = Signal(list)
    failed = Signal(str)

    def __init__(self, source: Any, count: int) -> None:
        super().__init__()
        self.source, self.count = source, count

    def run(self) -> None:
        try:
            self.done.emit(list(self.source.capture(self.count)))
        except RuntimeError as exc:
            self.failed.emit(f"Recording stopped: {exc}. Is another app using the camera?")


class ReconWorker(QObject):
    done = Signal(dict)

    def __init__(self, frames: list[np.ndarray], out: Path, sim: Any) -> None:
        super().__init__()
        self.frames, self.out, self.sim = frames, out, sim

    def run(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
        from ear_scan import analyse

        try:
            summary = analyse([np.asarray(f, np.float64) for f in self.frames], self.out, self.sim)
            summary["folder"] = str(self.out)
        except Exception as exc:  # noqa: BLE001 - report OpenCV failures plainly
            summary = {"error": f"Reconstruction failed: {exc}. Try a slower sweep with the wall in view.",
                       "folder": str(self.out)}
        self.done.emit(summary)


class LiveLabel(QLabel):
    def __init__(self, win: ScanWindow) -> None:
        super().__init__()
        self.win = win
        self.setFixedSize(VIEW_W, VIEW_H)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setStyleSheet("background:#111")

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        self.win.zoom_at(ev.position().x(), ev.position().y())


class ScanWindow(QMainWindow):
    def __init__(self, source: Any, simulated: bool = False) -> None:
        super().__init__()
        self.source, self.simulated = source, simulated
        self.frame_grey: np.ndarray | None = None
        self.frame_rgb: np.ndarray | None = None
        self.zoom_on = False
        self.zoom_point: tuple[float, float] | None = None
        self.countdown_end: float | None = None
        self.recording_until: float | None = None
        self.last_folder: Path | None = None
        self.session_folder = OUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
        self.recon_thread: QThread | None = None
        self.setWindowTitle("ECZIS Ear / Canal Scan (experimental)")

        root = QWidget()
        grid = QGridLayout(root)
        self.live = LiveLabel(self)
        grid.addWidget(self.live, 0, 0, 3, 1)
        self.status = QLabel("Starting camera...")
        self.status.setStyleSheet("font-size:15px;font-weight:bold;padding:4px")
        grid.addWidget(self.status, 3, 0)

        side = QVBoxLayout()
        self.rec_btn = QPushButton("●  Record sweep")
        self.rec_btn.setStyleSheet("font-size:16px;padding:10px;background:#b91c1c;color:white")
        self.rec_btn.clicked.connect(self.start_countdown)
        self.secs = QDoubleSpinBox()
        self.secs.setRange(3, 20)
        self.secs.setValue(8)
        self.secs.setSuffix(" s sweep")
        self.snap_btn = QPushButton("📷  Snapshot")
        self.snap_btn.setStyleSheet("font-size:15px;padding:8px")
        self.snap_btn.clicked.connect(self.snapshot)
        self.zoom_btn = QPushButton("🔍  Ultra-zoom: OFF")
        self.zoom_btn.setStyleSheet("font-size:15px;padding:8px")
        self.zoom_btn.clicked.connect(self.toggle_zoom)
        self.zoom_view = QLabel("Ultra-zoom: turn it on, then click the live image")
        self.zoom_view.setFixedSize(ZOOM_SIDE, ZOOM_SIDE)
        self.zoom_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_view.setStyleSheet("background:#111;color:#bbb")
        self.zoom_note = QLabel("")
        self.zoom_note.setWordWrap(True)
        self.open3d_btn = QPushButton("Open 3D scan (PLY)")
        self.open3d_btn.setEnabled(False)
        self.open3d_btn.clicked.connect(self.open_ply)
        self.folder_btn = QPushButton("Open results folder")
        self.folder_btn.clicked.connect(self.open_folder)
        for wdg in (self.rec_btn, self.secs, self.snap_btn, self.zoom_btn, self.zoom_view, self.zoom_note,
                    self.open3d_btn, self.folder_btn):
            side.addWidget(wdg)
        side.addStretch(1)
        grid.addLayout(side, 0, 1, 3, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(150)
        grid.addWidget(self.log, 4, 0, 1, 2)
        self.setCentralWidget(root)
        note = "SIMULATED canal (no camera)" if simulated else "Local only: images stay on this PC (artifacts/earscan)."
        self.say(note + " Output is uncalibrated / no scale; not a diagnosis.")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(40)

    # --- live -----------------------------------------------------------------
    def say(self, text: str) -> None:
        self.log.appendPlainText(text)

    def tick(self) -> None:
        try:
            grey = self.source.read()
        except RuntimeError:
            self.status.setText("Camera not delivering frames: close other camera apps")
            return
        colour = self.source.read_display() if hasattr(self.source, "read_display") else None
        self.frame_grey, self.frame_rgb = grey, colour
        show = colour if colour is not None else grey
        pm = to_pixmap(show, VIEW_W, VIEW_H)
        recording_until = getattr(self, "recording_until", None)
        if recording_until is not None:
            left = max(0.0, recording_until - time.monotonic())
            self.status.setText(f"● RECORDING {left:0.0f} s left - move slowly in and out  |  "
                                f"Motion {self.motion_label(grey)}")
            self.status.setStyleSheet("font-size:15px;font-weight:bold;padding:4px;color:#b91c1c")
        elif self.countdown_end is not None:
            left = self.countdown_end - time.monotonic()
            if left > 0:
                self.status.setText(f"RECORDING IN {left:0.0f}... get in position")
            else:
                self.countdown_end = None
                self.record()
                return
        else:
            self.status.setStyleSheet("font-size:15px;font-weight:bold;padding:4px")
            exp, foc, tip = guidance(grey)
            self.status.setText(f"Exposure {exp}  |  Focus {foc}  |  Motion {self.motion_label(grey)}  |  {tip}")
        self.live.setPixmap(pm)
        if self.zoom_on and self.zoom_point is not None:
            self.render_zoom()

    def motion_label(self, grey: np.ndarray) -> str:
        """Live movement readout: 3D needs the tip to move; a still probe sees no depth."""
        from eczis.lab.session import _quick_motion

        prev = getattr(self, "_prev_grey", None)
        self._prev_grey = grey
        if prev is None or prev.shape != grey.shape:
            return "-"
        self._motion = 0.8 * getattr(self, "_motion", 0.0) + 0.2 * _quick_motion(prev, grey)
        return "STILL (move slowly in/out)" if self._motion < 1.0 else ("TOO FAST" if self._motion > 25 else "GOOD")

    # --- actions ----------------------------------------------------------------
    def start_countdown(self) -> None:
        self.countdown_end = time.monotonic() + 3.0
        self.say(f"Recording {self.secs.value():.0f} s after a 3 s countdown...")

    def record(self) -> None:
        n = int(self.secs.value() * 30)
        self.recording_until = time.monotonic() + self.secs.value()
        self.rec_btn.setEnabled(False)
        self.say(f"Recording {self.secs.value():.0f} s: keep watching the live view and move slowly in and out.")
        self.capture_thread = QThread()
        self.capturer = CaptureWorker(self.source, n)
        self.capturer.moveToThread(self.capture_thread)
        self.capture_thread.started.connect(self.capturer.run)
        self.capturer.done.connect(self.capture_done)
        self.capturer.failed.connect(self.capture_failed)
        self.capturer.done.connect(self.capture_thread.quit)
        self.capturer.failed.connect(self.capture_thread.quit)
        self.capture_thread.start()

    def capture_failed(self, message: str) -> None:
        self.recording_until = None
        self.rec_btn.setEnabled(True)
        self.say(message)

    def capture_done(self, raw: list[np.ndarray]) -> None:
        self.recording_until = None
        frames = raw[::2]
        out = OUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "frames.npz", frames=np.stack(frames).astype(np.float32))
        self.last_folder = out
        self.say(f"Recorded {len(frames)} frames. Reconstructing in the background...")
        self.recon_thread = QThread()
        self.worker = ReconWorker(frames, out, getattr(self.source, "sim", None) if self.simulated else None)
        self.worker.moveToThread(self.recon_thread)
        self.recon_thread.started.connect(self.worker.run)
        self.worker.done.connect(self.recon_done)
        self.worker.done.connect(self.recon_thread.quit)
        self.recon_thread.start()

    def recon_done(self, summary: dict[str, Any]) -> None:
        self.rec_btn.setEnabled(True)
        if "error" in summary:
            self.say(summary["error"])
            return
        pts = summary.get("points", {})
        self.say(f"3D done: {summary.get('frames_registered')} frames placed; points R {pts.get('R')} / "
                 f"I {pts.get('I')} / U {pts.get('U')}; wall coverage {summary.get('coverage')}.")
        self.say(f"Next: {summary.get('advice')}")
        self.open3d_btn.setEnabled(True)
        (Path(summary["folder"]) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def snapshot(self) -> None:
        img = self.frame_rgb if self.frame_rgb is not None else self.frame_grey
        if img is None:
            return
        self.session_folder.mkdir(parents=True, exist_ok=True)
        path = self.session_folder / f"snapshot_{time.strftime('%H%M%S')}_{int(time.monotonic() * 1000) % 1000:03d}.png"
        from PIL import Image

        arr = img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
        self.last_folder = self.session_folder
        self.say(f"Snapshot saved: {path.name} ({arr.shape[1]}x{arr.shape[0]}, full resolution)")

    def toggle_zoom(self) -> None:
        self.zoom_on = not self.zoom_on
        self.zoom_btn.setText(f"🔍  Ultra-zoom: {'ON (click the image)' if self.zoom_on else 'OFF'}")
        if not self.zoom_on:
            self.zoom_view.setText("Ultra-zoom: turn it on, then click the live image")
            self.zoom_note.setText("")

    def zoom_at(self, x: float, y: float) -> None:
        if not self.zoom_on or self.frame_grey is None:
            return
        h, w = self.frame_grey.shape
        scale = min(VIEW_W / w, VIEW_H / h)
        self.zoom_point = (min(max(x / scale, 0), w - 1), min(max(y / scale, 0), h - 1))
        self.render_zoom()

    def render_zoom(self) -> None:
        if self.zoom_point is None:
            return
        img = self.frame_rgb if self.frame_rgb is not None else self.frame_grey
        if img is None:
            return
        h, w = img.shape[:2]
        half = 20  # 40x40 sensor pixels -> 8x magnification in a 320 px panel
        cx, cy = int(self.zoom_point[0]), int(self.zoom_point[1])
        x0, y0 = max(0, min(w - 2 * half, cx - half)), max(0, min(h - 2 * half, cy - half))
        crop = img[y0 : y0 + 2 * half, x0 : x0 + 2 * half]
        self.zoom_view.setPixmap(to_pixmap(crop, ZOOM_SIDE, ZOOM_SIDE, smooth=False))
        self.zoom_note.setText(f"8x at ({cx}, {cy}). Each block is one camera pixel: digital zoom shows the "
                               "pixels larger but adds no detail the sensor did not capture.")

    def open_ply(self) -> None:
        if self.last_folder and (self.last_folder / "scan.ply").exists():
            os.startfile(self.last_folder / "scan.ply")  # noqa: S606

    def open_folder(self) -> None:
        folder = self.last_folder or OUT_ROOT
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)  # noqa: S606

    def closeEvent(self, ev: Any) -> None:  # noqa: N802
        self.timer.stop()
        # A QThread destroyed while running crashes the process; finish capture/reconstruction first.
        for t in (getattr(self, "capture_thread", None), self.recon_thread):
            if t is not None and t.isRunning():
                t.quit()
                t.wait(20000)
        if hasattr(self.source, "close"):
            self.source.close()
        super().closeEvent(ev)


def build(args: argparse.Namespace) -> ScanWindow:
    if args.simulate:
        return ScanWindow(SyntheticSource(), simulated=True)
    import cv2

    from eczis.lab.sources import OpenCVCamera

    cam = OpenCVCamera(args.camera, cv2.CAP_MSMF)
    cam.set_mode(640, 480, "YUY2")
    cam.start()
    return ScanWindow(cam)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ECZIS Ear / Canal Scan (experimental)")
    ap.add_argument("--camera", type=int, default=None, help="device index; a real camera opens only when given")
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args(argv)
    if args.camera is None and not args.simulate:
        ap.error("give --camera <index> to open a device, or --simulate for the synthetic canal")
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        win = build(args)
    except RuntimeError as exc:
        print(f"Camera {args.camera} could not be opened ({exc}). Close other camera apps and retry.")
        return 2
    win.resize(1020, 760)
    win.show()
    win.raise_()
    win.activateWindow()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
