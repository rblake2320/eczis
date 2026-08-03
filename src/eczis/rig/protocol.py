# UNFILED-IP
from __future__ import annotations

from typing import Protocol

from eczis.rig.schedule import CaptureSchedule
from eczis.schemas import Calibration, CaptureSet


class CaptureRig(Protocol):
    def get_calibration(self) -> Calibration: ...

    def capture_burst(self, schedule: CaptureSchedule, *, root_seed: int) -> CaptureSet: ...
