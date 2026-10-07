# UNFILED-IP
"""Drive the Ear / Canal Scan window offscreen on the synthetic canal."""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

import eczis.lab.scan_app as sa  # noqa: E402


@pytest.fixture()
def win(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sa, "OUT_ROOT", tmp_path)
    QApplication.instance() or QApplication([])
    w = sa.ScanWindow(sa.SyntheticSource(), simulated=True)
    w.session_folder = tmp_path / "session"
    w.tick()
    yield w
    w.close()


def test_snapshot_saves_full_resolution_colour(win, tmp_path) -> None:  # type: ignore[no-untyped-def]
    win.snapshot()
    files = list((tmp_path / "session").glob("snapshot_*.png"))
    from PIL import Image

    assert len(files) == 1 and Image.open(files[0]).size == (320, 240) and Image.open(files[0]).mode == "RGB"


def test_ultra_zoom_click_shows_magnified_crop_with_honest_note(win) -> None:  # type: ignore[no-untyped-def]
    win.zoom_at(100, 100)  # zoom off: no effect
    assert win.zoom_point is None
    win.toggle_zoom()
    win.zoom_at(300, 200)
    assert win.zoom_point is not None and not win.zoom_view.pixmap().isNull()
    assert "adds no detail" in win.zoom_note.text()


def test_record_reconstructs_in_background_and_enables_3d(win) -> None:  # type: ignore[no-untyped-def]
    win.secs.setValue(3)  # 90 frames requested -> 60 synthetic frames, every 2nd kept
    win.record()
    assert not win.rec_btn.isEnabled()
    t0 = time.monotonic()
    while not win.rec_btn.isEnabled() and time.monotonic() - t0 < 120:
        QApplication.processEvents()
        time.sleep(0.05)
    log = win.log.toPlainText()
    assert "3D done" in log and win.open3d_btn.isEnabled(), log
    assert (win.last_folder / "scan.ply").exists() and (win.last_folder / "summary.json").exists()


def test_guidance_flags_glare_and_darkness() -> None:
    import numpy as np

    assert sa.guidance(np.ones((10, 10), np.float32))[0] == "GLARE"
    assert sa.guidance(np.zeros((10, 10), np.float32))[0] == "TOO DARK"
