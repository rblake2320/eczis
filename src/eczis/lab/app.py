# UNFILED-IP
"""ECZIS Microscope Lab desktop app (PySide6). EXPERIMENTAL.

    uv run python -m eczis.lab.app --simulated [--glyph 8|3]
    uv run python -m eczis.lab.app --camera 0 --profile profile.json   # real microscope

A real camera is only opened when --camera is given explicitly. Without a calibration profile
the analysis refuses (UNCALIBRATED). Focus Stack and Compare are planned, not implemented.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from eczis.lab.commands import ControlLevel, LoggingController, parse
from eczis.lab.pipeline import (
    ROI,
    Analysis,
    Burst,
    CalibrationProfile,
    analyze,
    investigate,
    register,
)
from eczis.lab.sources import SimulatedMicroscope, glyph_scene

VIEW = 384  # live view side in screen pixels
COLORS = {2: (37, 99, 235), 1: (245, 158, 11), 0: (60, 60, 60)}  # R blue, I amber, U grey


def to_pixmap(img: np.ndarray, side: int) -> QPixmap:
    a = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if a.ndim == 2:
        q = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format.Format_Grayscale8)
    else:
        q = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q.copy()).scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.FastTransformation)


def quality(frames: list[np.ndarray]) -> dict[str, str]:
    from scipy.ndimage import laplace

    f = frames[-1]
    focus = float(np.var(laplace(f.astype(np.float64))))
    clipped = float(np.mean((f <= 0.002) | (f >= 0.998)))
    motion = 0.0
    if len(frames) >= 2:
        shifts = register([frames[-2], frames[-1]])
        motion = float(np.hypot(shifts[1][0] - shifts[0][0], shifts[1][1] - shifts[0][1]))
    return {
        "Focus": "GOOD" if focus > 2e-4 else "SOFT",
        "Motion": "LOW" if motion < 0.5 else "HIGH",
        "Exposure": "GOOD" if clipped < 0.01 else "CLIPPING",
    }


class LiveView(QLabel):
    """Live image; in Mark Region mode a drag selects the feature rectangle."""

    def __init__(self, window: LabWindow) -> None:
        super().__init__()
        self.window_ref = window
        self.setFixedSize(VIEW, VIEW)
        self.setStyleSheet("background:#111")
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)  # click mapping origin
        self.marking = False
        self._start: QPoint | None = None
        self.rect_screen: QRect | None = None

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if self.marking:
            self._start = ev.position().toPoint()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if self.marking and self._start is not None:
            self.rect_screen = QRect(self._start, ev.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if self.marking and self.rect_screen is not None:
            self.marking = False
            self.window_ref.region_marked(self.rect_screen)

    def paintEvent(self, ev: Any) -> None:  # noqa: N802
        super().paintEvent(ev)
        if self.rect_screen is not None:
            p = QPainter(self)
            p.setPen(QPen(Qt.GlobalColor.yellow, 2))
            p.drawRect(self.rect_screen)
            p.end()


class LabWindow(QMainWindow):
    def __init__(self, source: Any, profile: CalibrationProfile | None, controller: Any) -> None:
        super().__init__()
        self.source, self.profile, self.controller = source, profile, controller
        self.recent: list[np.ndarray] = []
        self.bursts: list[Burst] = []
        self.roi_hr: tuple[int, int] | None = None
        self.feature_mask: np.ndarray | None = None
        self.last: Analysis | None = None
        self.setWindowTitle("ECZIS Microscope Lab (experimental)")

        root = QWidget()
        grid = QGridLayout(root)
        self.header = QLabel()
        self.header.setStyleSheet("font-weight:bold;padding:4px")
        grid.addWidget(self.header, 0, 0, 1, 2)

        left = QVBoxLayout()
        self.view = LiveView(self)
        left.addWidget(self.view)
        self.status = QLabel("Focus: -   Motion: -   Exposure: -")
        left.addWidget(self.status)
        grid.addLayout(left, 1, 0)

        right = QVBoxLayout()
        qbox = QGroupBox("Question")
        ql = QVBoxLayout(qbox)
        self.question = QComboBox()
        self.question.setEditable(False)
        self.question.addItems(["Is the marked feature present?"])
        ql.addWidget(self.question)
        self.result = QLabel("Current result: -")
        self.result.setStyleSheet("font-size:14px;font-weight:bold")
        ql.addWidget(self.result)
        self.interval = QLabel("")
        ql.addWidget(self.interval)
        self.suggestion = QLabel("Suggested action: -")
        self.suggestion.setWordWrap(True)
        ql.addWidget(self.suggestion)
        lv = QHBoxLayout()
        lv.addWidget(QLabel("Control level"))
        self.level = QComboBox()
        self.level.addItems([lvl.name for lvl in ControlLevel])
        self.level.setCurrentIndex(int(ControlLevel.MANUAL))
        lv.addWidget(self.level)
        ql.addLayout(lv)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.investigate_btn = QPushButton("Investigate (run loop at control level)")
        self.investigate_btn.clicked.connect(self.run_investigation)
        ql.addWidget(self.analyze_btn)
        ql.addWidget(self.investigate_btn)
        right.addWidget(qbox)
        self.ranking = QPlainTextEdit()
        self.ranking.setReadOnly(True)
        self.ranking.setFixedHeight(120)
        right.addWidget(self.ranking)
        grid.addLayout(right, 1, 1)

        buttons = QGridLayout()
        self.capture_btn = QPushButton("Capture Burst")
        self.capture_btn.clicked.connect(self.capture_burst)
        self.stack_btn = QPushButton("Focus Stack")
        self.stack_btn.setEnabled(False)
        self.stack_btn.setToolTip("Planned: adaptive focus stack")
        self.mark_btn = QPushButton("Mark Region")
        self.mark_btn.clicked.connect(self.start_marking)
        self.compare_btn = QPushButton("Compare")
        self.compare_btn.setEnabled(False)
        self.compare_btn.setToolTip("Planned: certified change between sessions")
        for i, b in enumerate((self.capture_btn, self.stack_btn, self.mark_btn, self.compare_btn)):
            buttons.addWidget(b, i // 2, i % 2)
        self.demo_btn = QPushButton("Use demo region")
        self.demo_btn.clicked.connect(self.use_demo_region)
        buttons.addWidget(self.demo_btn, 2, 0)
        cmd_row = QHBoxLayout()
        self.cmd = QLineEdit()
        self.cmd.setPlaceholderText("Command, e.g. FOCUS +12 / LIGHT SIDE_LEFT 70 / ZOOM 2 / CAPTURE 16")
        send = QPushButton("Send")
        send.clicked.connect(self.send_command)
        cmd_row.addWidget(self.cmd)
        cmd_row.addWidget(send)
        bl = QVBoxLayout()
        bl.addLayout(buttons)
        bl.addLayout(cmd_row)
        grid.addLayout(bl, 2, 0)

        ev = QGroupBox("Evidence layers")
        el = QHBoxLayout(ev)
        evc = QVBoxLayout()
        self.layers = {name: QCheckBox(name) for name in ("Supported", "Inferred", "Unsupported")}
        for cb in self.layers.values():
            cb.setChecked(True)
            cb.stateChanged.connect(self.refresh_evidence)
            evc.addWidget(cb)
        el.addLayout(evc)
        self.evidence = QLabel()
        self.evidence.setFixedSize(192, 192)
        self.evidence.setStyleSheet("background:#111")
        el.addWidget(self.evidence)
        grid.addWidget(ev, 2, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(90)
        grid.addWidget(self.log, 3, 0, 1, 2)
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)
        self.update_header()
        if profile is None:
            self.result.setText("Current result: UNCALIBRATED (no calibration profile)")

    # --- live --------------------------------------------------------------
    def update_header(self) -> None:
        prof = self.profile.source if self.profile else "NONE"
        self.header.setText(f"{self.source.describe()}    Connected    Calibration: {prof}")

    def tick(self) -> None:
        frame = self.source.read()
        self.recent = (self.recent + [frame])[-2:]
        self.view.setPixmap(to_pixmap(frame, VIEW))
        q = quality(self.recent)
        self.status.setText("   ".join(f"{k}: {v}" for k, v in q.items()))

    def say(self, text: str) -> None:
        self.log.appendPlainText(text)

    # --- actions -----------------------------------------------------------
    def capture_burst(self) -> None:
        frames = self.source.capture(16)
        self.bursts.append(Burst(list(frames), int(getattr(self.source, "zoom", 1)), self.source.geometry()))
        self.say(f"Captured burst of {len(frames)} frames at zoom {self.bursts[-1].zoom}x")

    def start_marking(self) -> None:
        self.view.marking = True
        self.view.rect_screen = None
        self.say("Mark Region: drag a rectangle around the feature in the live view")

    def region_marked(self, rect: QRect) -> None:
        geo = self.source.geometry()
        h, w = self.recent[-1].shape
        px_per_sensor = VIEW / max(h, w)  # pixmap keeps aspect ratio, anchored top-left

        def to_hr(sx: float, sy: float) -> tuple[float, float]:
            return (geo.origin_hr[0] + sy / px_per_sensor * geo.scale, geo.origin_hr[1] + sx / px_per_sensor * geo.scale)

        (y0, x0), (y1, x1) = to_hr(rect.left(), rect.top()), to_hr(rect.right(), rect.bottom())
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        snap = geo.scale * 4  # ROI aligned with every supported zoom's sensor grid
        ry = int(round((cy - ROI / 2) / snap) * snap)
        rx = int(round((cx - ROI / 2) / snap) * snap)
        self.roi_hr = (ry, rx)
        mask = np.zeros((ROI, ROI), bool)
        mask[max(0, int(y0) - ry) : min(ROI, int(np.ceil(y1)) - ry), max(0, int(x0) - rx) : min(ROI, int(np.ceil(x1)) - rx)] = True
        self.feature_mask = mask
        self.say(f"Region set: ROI at {self.roi_hr}, feature mask {int(mask.sum())} px")

    def use_demo_region(self) -> None:
        self.roi_hr = (48, 48)
        m = np.zeros((ROI, ROI), bool)
        m[11:23, 10:12] = True
        self.feature_mask = m
        self.say("Demo region: left stroke of the glyph (the '8' vs '3' difference)")

    def send_command(self) -> None:
        try:
            cmd = parse(self.cmd.text())
        except ValueError as exc:
            self.say(f"Rejected: {exc}")
            return
        reply = self.controller.send(cmd)
        self.say(f"> {cmd.encode()}  <- {reply}")
        if cmd.verb == "CAPTURE" and getattr(self.source, "last_burst", None):
            self.bursts.append(Burst(list(self.source.last_burst), int(getattr(self.source, "zoom", 1)), self.source.geometry()))
        self.update_header()

    def _ready(self) -> bool:
        if self.roi_hr is None or self.feature_mask is None:
            self.say("Mark a region first (or use the demo region)")
            return False
        return True

    def run_analysis(self) -> None:
        if not self._ready():
            return
        if not self.bursts:
            self.capture_burst()
        assert self.roi_hr is not None and self.feature_mask is not None
        try:
            zooms = self.profile.zoom_levels if self.profile else (1,)
            self.show_result(analyze(self.bursts, self.profile, self.roi_hr, self.feature_mask, zooms))
        except ValueError as exc:
            self.say(f"Analysis refused: {exc}")

    def run_investigation(self) -> None:
        if not self._ready() or self.profile is None:
            self.say("Investigation needs a calibration profile and a marked region")
            return
        assert self.roi_hr is not None and self.feature_mask is not None
        level = ControlLevel[self.level.currentText()]
        steps = investigate(self.source, self.controller, self.profile, self.roi_hr, self.feature_mask, level)
        for s in steps:
            self.say(f"zoom {s.zoom}x: {s.analysis.status}" + (f"  -> executed {', '.join(s.executed)}" if s.executed else ""))
        self.update_header()
        self.show_result(steps[-1].analysis)

    def show_result(self, result: Analysis) -> None:
        self.last = result
        self.result.setText(f"Current result: {result.status}")
        c = result.certificate
        self.interval.setText("" if c is None else f"Answer range [{c.lower:.3f}, {c.upper:.3f}]")
        s = result.suggestion
        if s is None:
            self.suggestion.setText("Suggested action: none needed" if c and c.decision.value != "undetermined" else "Suggested action: -")
            self.ranking.setPlainText("\n".join(f"{k}: {v}" for k, v in result.band_summary.items()))
        else:
            self.suggestion.setText(f"Suggested action: {s.action}  [{' ; '.join(s.commands)}]  predicted gain {s.gain:.0%}")
            lines = [f"{name}: answer s.d. {sd:.4f}" for name, sd in s.ranking]
            self.ranking.setPlainText("Capture options (lower is better):\n" + "\n".join(lines) + "\n\nAssumptions:\n- " + "\n- ".join(result.assumptions))
        self.refresh_evidence()

    def refresh_evidence(self) -> None:
        if self.last is None or self.last.support is None:
            return
        f = self.last.support
        img = f.mean.reshape(f.shape)
        cls = f.pixel_class(1)
        rgb = np.repeat(np.clip(img, 0, 1)[..., None], 3, axis=2)
        for code, name in ((2, "Supported"), (1, "Inferred"), (0, "Unsupported")):
            if self.layers[name].isChecked():
                sel = cls == code
                rgb[sel] = 0.55 * rgb[sel] + 0.45 * np.array(COLORS[code]) / 255.0
        self.evidence.setPixmap(to_pixmap(rgb, 192))


def load_profile(path: Path | None) -> CalibrationProfile | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["zoom_levels"] = tuple(data.get("zoom_levels", (1,)))
    return CalibrationProfile(**data)


def build_window(args: argparse.Namespace) -> LabWindow:
    if args.camera is not None:
        import cv2

        from eczis.lab.sources import OpenCVCamera

        backend = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}[args.backend]
        camera = OpenCVCamera(args.camera, backend)
        if args.size:
            w, h = (int(v) for v in args.size.lower().split("x"))
            camera.set_mode(w, h, args.fourcc)
        return LabWindow(camera, load_profile(args.profile), LoggingController())
    sim = SimulatedMicroscope(glyph_scene(args.glyph == "8"), seed=args.seed)
    profile = load_profile(args.profile) or CalibrationProfile(0.70, 0.004, 700.0, zoom_levels=(1, 2, 4), source="simulated")
    return LabWindow(sim, profile, sim)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ECZIS Microscope Lab (experimental)")
    ap.add_argument("--simulated", action="store_true", help="use the simulated microscope (default)")
    ap.add_argument("--glyph", choices=["8", "3"], default="8")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--camera", type=int, default=None, help="open a real camera by index (explicit opt-in)")
    ap.add_argument("--backend", choices=["msmf", "dshow"], default="msmf")
    ap.add_argument("--size", default=None, help="capture mode, e.g. 640x480")
    ap.add_argument("--fourcc", default="YUY2", help="pixel format; uncompressed YUY2 preferred over MJPG")
    ap.add_argument("--profile", type=Path, default=None)
    args = ap.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    win = build_window(args)
    win.resize(900, 760)
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
