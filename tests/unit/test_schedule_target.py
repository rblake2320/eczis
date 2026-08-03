# UNFILED-IP
from __future__ import annotations

import numpy as np
import pytest

from eczis.rig.schedule import CaptureSchedule, lattice_schedule
from eczis.schemas import ObservationParams
from eczis.targets.microglyph import create_microglyph_target


def test_lattice_schedule_has_distinct_phases() -> None:
    schedule = lattice_schedule(16, 7)
    phases = {(round(x.dx_lr, 5), round(x.dy_lr, 5)) for x in schedule.observations}
    assert len(phases) == 16


def test_degenerate_schedule_fails_closed() -> None:
    obs = ObservationParams("same", 0, 0, 0, 1, 0, 0, 1)
    schedule = CaptureSchedule((obs, obs, obs, obs))
    with pytest.raises(ValueError, match="distinct subpixel"):
        schedule.validate()


def test_microglyph_contains_subsensor_features() -> None:
    target = create_microglyph_target((96, 128), seed=9, stroke_px=1)
    assert target.feature_count > 100
    assert target.feature_mask.dtype == np.bool_
    # One HR-pixel strokes are smaller than a 4x sensor-pixel footprint.
    assert int(target.feature_mask.sum()) > 0
