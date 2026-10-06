# UNFILED-IP
"""Drive the Microscope Lab window end to end (offscreen by default; set QT_QPA_PLATFORM to override)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from eczis.lab.app import build_window, main  # noqa: E402
from eczis.lab.commands import ControlLevel  # noqa: E402


def _window(glyph: str):  # type: ignore[no-untyped-def]
    import argparse

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    return build_window(argparse.Namespace(camera=None, profile=None, glyph=glyph, seed=3, backend="msmf", size=None, fourcc="YUY2"))


@pytest.mark.parametrize("glyph,expected", [("8", "CERTIFIED: feature present"), ("3", "CERTIFIED: feature absent")])
def test_window_investigation_reaches_certificate(glyph: str, expected: str) -> None:
    win = _window(glyph)
    win.tick()
    win.tick()
    assert "Focus:" in win.status.text()
    win.use_demo_region()
    win.level.setCurrentIndex(int(ControlLevel.AUTONOMOUS))
    win.run_investigation()
    assert expected in win.result.text()
    assert "Answer range" in win.interval.text()
    assert win.evidence.pixmap() is not None and not win.evidence.pixmap().isNull()
    assert "executed ZOOM 4, CAPTURE 16" in win.log.toPlainText()


def test_manual_level_shows_instruction_without_moving_hardware() -> None:
    win = _window("8")
    win.tick()
    win.use_demo_region()
    win.run_investigation()  # default level MANUAL
    assert "UNDETERMINED" in win.result.text()
    assert "ZOOM 4" in win.suggestion.text()
    assert "digital zoom (adds no measurement)" in win.ranking.toPlainText()
    assert win.source.zoom == 1


def test_command_box_validates_and_planned_buttons_are_disabled() -> None:
    win = _window("8")
    win.cmd.setText("DIGITAL_ZOOM 4")
    win.send_command()
    assert "Rejected" in win.log.toPlainText()
    win.cmd.setText("LIGHT SIDE_LEFT 70")
    win.send_command()
    assert "OK LIGHT SIDE_LEFT 70" in win.log.toPlainText()
    assert not win.stack_btn.isEnabled() and not win.compare_btn.isEnabled()


def test_main_rejects_bad_arguments() -> None:
    with pytest.raises(SystemExit):
        main(["--glyph", "7"])


def test_region_marking_maps_non_square_frames() -> None:
    """A 640x480-style (non-square) frame maps clicks with one scale anchored top-left."""
    import numpy as np
    from PySide6.QtCore import QRect

    win = _window("8")
    win.recent = [np.zeros((48, 64), np.float32)]  # 4:3 frame
    win.region_marked(QRect(120, 120, 12, 24))  # screen px -> 64-px-wide frame at 384 px
    assert win.roi_hr is not None and win.feature_mask is not None
    assert int(win.feature_mask.sum()) > 0
