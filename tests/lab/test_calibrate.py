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


def test_profile_records_per_view_errors_and_identity(tmp_path):
    import numpy as np

    from eczis.lab.calibrate import Calibrator, LensProfile, summary, synthetic_views

    k = np.array([[520.0, 0, 322.0], [0, 518.0, 238.0], [0, 0, 1.0]])
    d = np.array([-0.28, 0.09, 0.0005, -0.0004, 0.0])
    cal = Calibrator()
    for v in synthetic_views(k, d, (640, 480), count=24, seed=3):
        cal.add(v)
    prof = cal.solve(camera="camera 1 (synthetic)", capture_mode="640x480 YUY2 MSMF")
    assert len(prof.per_view_rms_px) == prof.views >= 8
    assert all(e >= 0 for e in prof.per_view_rms_px)
    assert max(prof.per_view_rms_px) < 2.0  # no single view dominates on clean synthetic data
    s = summary(prof)
    assert s["worst_view_rms_px"] == round(max(prof.per_view_rms_px), 3)
    back = LensProfile.from_json(prof.to_json(tmp_path / "lens.json"))
    assert back == prof and back.camera == "camera 1 (synthetic)" and back.capture_mode == "640x480 YUY2 MSMF"
    assert json_has_fields(tmp_path / "lens.json")


def json_has_fields(path) -> bool:
    import json

    d = json.loads(path.read_text(encoding="utf-8"))
    return {"per_view_rms_px", "camera", "capture_mode", "model", "created"} <= set(d)


def test_old_format_profile_without_new_fields_loads(tmp_path):
    """Profiles written before per-view errors / identity were recorded must still load."""
    import json

    from eczis.lab.calibrate import LensProfile

    old = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "dist": [0.1, -0.2, 0.0, 0.0, 0.0], "width": 640, "height": 480,
           "rms_px": 0.4, "views": 10, "square_mm": 2.0, "source": "checkerboard", "created": "2026-10-06T00:00:00"}
    p = tmp_path / "old.json"
    p.write_text(json.dumps(old), encoding="utf-8")
    prof = LensProfile.from_json(p)
    assert prof.per_view_rms_px == () and prof.camera == "" and prof.capture_mode == "" and prof.ok()
