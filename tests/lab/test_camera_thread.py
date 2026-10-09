# UNFILED-IP
"""The real-camera adapter must never block the caller on a stalled driver (regression: the Lab
window went "Not Responding" because cap.read() ran on the UI thread)."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("cv2")

from eczis.lab.sources import OpenCVCamera  # noqa: E402


class FakeCap:
    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        self.delay, self.fail, self.n = delay, fail, 0
        self.released = threading.Event()

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2 API
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(self.delay)
        if self.fail:
            return False, None
        self.n += 1
        return True, np.full((48, 64, 3), self.n % 255, np.uint8)

    def release(self) -> None:
        self.released.set()


def test_read_never_blocks_on_a_stalled_driver() -> None:
    cam = OpenCVCamera(1, cap=FakeCap(delay=5.0))
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError):
        cam.read()  # driver is stuck in a 5 s read; caller must not wait for it
    assert time.perf_counter() - t0 < 0.5
    cam._stop.set()


def test_streaming_frames_and_burst_are_distinct_and_grey() -> None:
    cam = OpenCVCamera(1, cap=FakeCap(delay=0.01))
    first = cam.wait_for_frame(2.0)
    assert first.shape == (48, 64) and first.dtype == np.float32
    burst = cam.capture(5)
    assert len(burst) == 5 and len({float(f[0, 0]) for f in burst}) == 5  # five distinct frames
    cam.close()


def test_failing_camera_reports_reason_quickly() -> None:
    cam = OpenCVCamera(1, cap=FakeCap(fail=True))
    with pytest.raises(RuntimeError, match="camera read failed"):
        cam.wait_for_frame(1.0)
    with pytest.raises(RuntimeError):
        cam.capture(2, timeout_per_frame=0.3)
    cam.close()


class FlakyCap(FakeCap):
    """Fails every other read, like a Media Foundation stream warming up."""

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(0.01)
        self.n += 1
        if self.n % 2:
            return False, None
        return True, np.full((48, 64, 3), self.n % 255, np.uint8)


def test_transient_read_failures_do_not_stop_the_live_view() -> None:
    cam = OpenCVCamera(1, cap=FlakyCap())
    cam.wait_for_frame(2.0)
    for _ in range(20):  # a live view polling at ~50 Hz must always get a frame
        assert cam.read().shape == (48, 64)
        time.sleep(0.02)
    cam.close()


def test_stale_stream_is_reported() -> None:
    cap = FakeCap(delay=0.01)
    cam = OpenCVCamera(1, cap=cap)
    cam.wait_for_frame(2.0)
    cap.fail = True  # stream stops delivering
    time.sleep(0.3)
    with pytest.raises(RuntimeError):
        cam.read(max_age=0.2)
    cam.close()


def test_colour_frame_kept_for_display_grey_for_analysis() -> None:
    class ColourCap(FakeCap):
        def read(self) -> tuple[bool, np.ndarray | None]:
            time.sleep(0.01)
            f = np.zeros((48, 64, 3), np.uint8)
            f[..., 2] = 200  # BGR: pure red
            return True, f

    cam = OpenCVCamera(1, cap=ColourCap())
    grey = cam.wait_for_frame(2.0)
    rgb = cam.read_display()
    assert grey.ndim == 2 and rgb is not None and rgb.shape == (48, 64, 3)
    assert rgb[0, 0, 0] == 200 and rgb[0, 0, 2] == 0  # converted BGR -> RGB for display
    cam.close()
