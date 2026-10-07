# UNFILED-IP
"""Frame sources: a simulated microscope (for development and tests) and a real camera adapter.

The simulated microscope obeys the same command protocol as hardware. It renders a known scene
through the estimated camera model with zero-mean hand/stage vibration, Poisson + read noise and
8-bit quantisation, so the whole loop can be exercised without a physical device.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.lab.commands import Command
from eczis.schemas import ObservationParams

F32 = npt.NDArray[np.float32]

BASE_SCALE = 4  # reconstruction-grid pixels per sensor pixel at zoom 1
FRAME_LR = 32  # simulated sensor frame side (sensor pixels)
SCENE_HR = 128  # simulated scene side (reconstruction-grid pixels)


@dataclass(frozen=True, slots=True)
class Geometry:
    """Maps sensor pixels to reconstruction-grid coordinates for the current optical state."""

    scale: int  # reconstruction-grid pixels per sensor pixel
    origin_hr: tuple[float, float]  # reconstruction-grid (y, x) of sensor pixel (0, 0)'s corner


def glyph_scene(eight: bool, contrast: float = 0.5, texture: float = 0.03, seed: int = 7) -> np.ndarray:
    """A 128x128 scene with a 2-px-stroke '8' or '3' centred in a 32x32 region at (48, 48).

    Faint smoothed paper-like texture gives registration something to lock onto, as real
    specimens do; set texture=0 for a flat page.
    """
    from scipy.ndimage import gaussian_filter

    grain = gaussian_filter(np.random.default_rng(seed).standard_normal((SCENE_HR, SCENE_HR)), 2.0)
    x = 0.2 + texture * grain / max(float(np.abs(grain).max()), 1e-9)
    r0, c0 = 48, 48
    for r in (8, 16, 24):
        x[r0 + r : r0 + r + 2, c0 + 10 : c0 + 22] += contrast
    x[r0 + 8 : r0 + 26, c0 + 20 : c0 + 22] += contrast
    if eight:
        x[r0 + 8 : r0 + 26, c0 + 10 : c0 + 12] += contrast
    return np.asarray(np.clip(x, 0.0, 1.0), dtype=np.float64)


@dataclass
class SimulatedMicroscope:
    scene: np.ndarray
    psf_sigma_lr: float = 0.70
    read_noise: float = 0.004
    shot_electrons: float = 700.0
    vibration_lr: float = 0.25  # s.d. of frame-to-frame jitter in sensor pixels
    zoom_levels: tuple[int, ...] = (1, 2, 4)
    seed: int = 0
    zoom: int = 1
    focus: int = 0  # 0 = best focus
    light: dict[str, int] = field(default_factory=lambda: {"RING": 70})
    stage: tuple[int, int] = (0, 0)
    last_burst: list[F32] = field(default_factory=list)
    true_jitter: list[tuple[float, float]] = field(default_factory=list)  # tests only
    _rng: np.random.Generator = field(init=False)
    _frame_no: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # --- geometry -----------------------------------------------------------
    def geometry(self) -> Geometry:
        scale = BASE_SCALE // self.zoom
        extent = FRAME_LR * scale
        centre = SCENE_HR / 2.0
        return Geometry(scale, (centre - extent / 2.0 + self.stage[0], centre - extent / 2.0 + self.stage[1]))

    def describe(self) -> str:
        return f"Simulated microscope  {FRAME_LR}x{FRAME_LR} sensor  zoom {self.zoom}x"

    # --- imaging ------------------------------------------------------------
    def _render(self, dx: float, dy: float) -> F32:
        geo = self.geometry()
        y0, x0 = int(round(geo.origin_hr[0])), int(round(geo.origin_hr[1]))
        extent = FRAME_LR * geo.scale
        field_hr = self.scene[y0 : y0 + extent, x0 : x0 + extent].astype(np.float32)
        psf = self.psf_sigma_lr + 0.05 * abs(self.focus)  # sensor-pixel blur; defocus widens it
        model = EstimatedCameraModel(scale=geo.scale, psf_sigma_lr=psf)
        exposure = max(self.light.values(), default=0) / 70.0
        params = ObservationParams(f"S{self._frame_no}", dx, dy, 0.0, exposure, 0.0, 0.0, 0)
        clean = np.clip(model.project(field_hr, params), 0.0, 1.0)
        shot = self._rng.poisson(clean * self.shot_electrons) / self.shot_electrons
        noisy = np.clip(shot + self._rng.normal(0.0, self.read_noise, clean.shape), 0.0, 1.0)
        self._frame_no += 1
        return np.asarray(np.round(noisy * 255.0) / 255.0, dtype=np.float32)  # 8-bit camera output

    def read(self) -> F32:
        dx, dy = self._rng.normal(0.0, self.vibration_lr, 2)
        return self._render(float(dx), float(dy))

    def capture(self, count: int) -> list[F32]:
        self.last_burst, self.true_jitter = [], []
        for _ in range(count):
            dx, dy = (float(v) for v in self._rng.normal(0.0, self.vibration_lr, 2))
            self.true_jitter.append((dx, dy))
            self.last_burst.append(self._render(dx, dy))
        return self.last_burst

    # --- controller ---------------------------------------------------------
    def send(self, command: Command) -> str:
        v = command.verb
        if v == "FOCUS":
            self.focus += int(command.args[0])
        elif v == "STAGE_X":
            self.stage = (self.stage[0], self.stage[1] + int(command.args[0]))
        elif v == "STAGE_Y":
            self.stage = (self.stage[0] + int(command.args[0]), self.stage[1])
        elif v == "LIGHT":
            self.light[str(command.args[0])] = int(command.args[1])
        elif v == "ZOOM":
            level = int(command.args[0])
            if level not in self.zoom_levels:
                return f"ERR ZOOM {level} unsupported"
            self.zoom = level
        elif v == "CAPTURE":
            self.capture(int(command.args[0]))
        return f"OK {command.encode()}"


class OpenCVCamera:
    """Real USB camera/microscope via OpenCV (Windows Media Foundation backend by default).

    Never started automatically: opening a camera captures whatever it points at.
    """

    def __init__(self, index: int, backend: int | None = None, cap: Any = None) -> None:
        import threading

        import cv2

        self._cv2 = cv2
        self.cap = cap if cap is not None else cv2.VideoCapture(index, cv2.CAP_MSMF if backend is None else backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} could not be opened")
        self.index = index
        # A background thread owns cap.read(): driver calls can block for seconds when a camera is
        # busy or stalls, and must never run on the UI thread.
        self._cond = threading.Condition()
        self._latest: F32 | None = None
        self._latest_rgb: np.ndarray | None = None  # display only; analysis uses grey intensity
        self._latest_t = 0.0
        self._seq = 0
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.zoom = 1  # fixed optics; magnification changes physically (distance, objective)

    def geometry(self) -> Geometry:
        """Reconstruction grid at BASE_SCALE x the sensor grid, anchored at the frame corner."""
        return Geometry(BASE_SCALE, (0.0, 0.0))

    def set_mode(self, width: int, height: int, fourcc: str = "YUY2") -> tuple[int, int, str]:
        """Request a capture mode (uncompressed YUY2 preferred); returns what the driver granted."""
        cv2 = self._cv2
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        got = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), got

    def describe(self) -> str:
        w = int(self.cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(self._cv2.CAP_PROP_FPS)
        return f"USB camera {self.index}  {w}x{h} @ {fps:.0f} fps"

    def probe_modes(self, sizes: tuple[tuple[int, int], ...] = ((640, 480), (1280, 720), (1920, 1080), (2592, 1944))) -> list[dict[str, Any]]:
        cv2 = self._cv2
        modes = []
        for fourcc in ("YUY2", "MJPG", "NV12"):
            for w, h in sizes:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                ok, frame = self.cap.read()
                got = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                modes.append({"requested": f"{fourcc} {w}x{h}", "read_ok": bool(ok), "got_size": got,
                              "got_fourcc": "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)),
                              "fps": self.cap.get(cv2.CAP_PROP_FPS)})
        return modes

    def lock_settings(self) -> dict[str, Any]:
        """Try to disable auto exposure / gain / white balance; report what the driver accepted."""
        cv2 = self._cv2
        before = {p: self.cap.get(getattr(cv2, p)) for p in ("CAP_PROP_AUTO_EXPOSURE", "CAP_PROP_EXPOSURE", "CAP_PROP_GAIN", "CAP_PROP_AUTO_WB")}
        accepted = {
            "auto_exposure_off": bool(self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)),
            "auto_wb_off": bool(self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)),
        }
        after = {p: self.cap.get(getattr(cv2, p)) for p in before}
        return {"before": before, "accepted": accepted, "after": after}

    def _grab_loop(self) -> None:
        import time

        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                with self._cond:
                    self._error = "camera read failed"
                time.sleep(0.2)
                continue
            gray = frame if frame.ndim == 2 else self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
            rgb = None if frame.ndim == 2 else self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            img = np.asarray(np.asarray(gray, dtype=np.float32) / np.float32(255.0), dtype=np.float32)
            with self._cond:
                self._latest, self._seq, self._error = img, self._seq + 1, None
                self._latest_rgb = rgb
                self._latest_t = time.monotonic()
                self._cond.notify_all()

    def start(self) -> None:
        import threading

        if self._thread is None:
            self._thread = threading.Thread(target=self._grab_loop, name=f"camera-{self.index}", daemon=True)
            self._thread.start()

    def wait_for_frame(self, timeout: float = 3.0) -> F32:
        """Block (off the UI thread) until a first frame arrives, or raise with the reason."""
        self.start()
        with self._cond:
            if self._latest is None:
                self._cond.wait_for(lambda: self._latest is not None, timeout)
            if self._latest is None:
                raise RuntimeError(self._error or "no frames within timeout")
            return self._latest.copy()

    def read(self, max_age: float = 2.0) -> F32:
        """Newest frame, never blocking. Transient driver read failures (common while a Media
        Foundation stream starts) are ignored while a frame younger than max_age seconds exists."""
        import time

        self.start()
        with self._cond:
            if self._latest is None or time.monotonic() - self._latest_t > max_age:
                raise RuntimeError(self._error or "no frames yet")
            return self._latest.copy()

    def read_display(self) -> np.ndarray | None:
        """Newest colour frame (uint8 RGB) for the live view, or None if the camera is mono."""
        with self._cond:
            return None if self._latest_rgb is None else self._latest_rgb.copy()

    def capture(self, count: int, timeout_per_frame: float = 2.0) -> list[F32]:
        """The next `count` new frames (distinct sequence numbers), each within a timeout."""
        self.start()
        frames: list[F32] = []
        with self._cond:
            seen = self._seq
            for _ in range(count):
                last = seen

                def newer(last: int = last) -> bool:
                    return self._seq > last

                if not self._cond.wait_for(newer, timeout_per_frame):
                    raise RuntimeError(self._error or "camera stopped delivering frames during burst")
                seen = self._seq
                assert self._latest is not None
                frames.append(self._latest.copy())
        return frames

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.cap.release()
