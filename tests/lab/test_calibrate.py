# UNFILED-IP
"""Lens calibration must recover a known synthetic lens and refuse to solve on too few views."""
from __future__ import annotations

import numpy as np
import pytest

from eczis.lab.calibrate import Calibrator, LensProfile, synthetic_views

K = np.array([[520.0, 0, 322.0], [0, 518.0, 238.0], [0, 0, 1.0]])
D = np.array([-0.28, 0.09, 0.0005, -0.0004, 0.0])


@pytest.fixture(scope="module")
def solved() -> LensProfile:
    cal = Calibrator(square_mm=2.0)
    for v in synthetic_views(K, D, (640, 480), count=30, seed=1):
        cal.add(v)
    assert len(cal.detections) >= 8
    return cal.solve()


def test_recovers_known_lens_within_tolerance(solved: LensProfile) -> None:
    assert abs(solved.fx - 520) / 520 < 0.03 and abs(solved.fy - 518) / 518 < 0.03
    assert abs(solved.cx - 322) < 8 and abs(solved.cy - 238) < 10  # 12 synthetic views: ~5 px typical
    assert solved.dist[0] < 0  # barrel sign recovered
    assert solved.rms_px < 1.0 and solved.ok()


def test_profile_round_trips_json(solved: LensProfile, tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = solved.to_json(tmp_path / "lens.json")
    back = LensProfile.from_json(p)
    assert back == solved
    assert back.matrix()[0, 0] == solved.fx


def test_refuses_with_too_few_views() -> None:
    cal = Calibrator()
    for v in synthetic_views(K, D, (640, 480), count=3, seed=2):
        cal.add(v)
    with pytest.raises(ValueError, match="at least 8"):
        cal.solve()


def test_duplicate_poses_are_not_counted_twice() -> None:
    cal = Calibrator()
    v = next(x for x in synthetic_views(K, D, (640, 480), count=20, seed=3) if cal.detect(x) is not None)
    assert cal.add(v) is True
    assert cal.add(v) is False  # same view again: rejected as near-duplicate
    assert len(cal.detections) == 1
