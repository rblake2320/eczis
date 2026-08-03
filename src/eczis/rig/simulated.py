# UNFILED-IP
from __future__ import annotations

import hashlib

from eczis.camera.truth_model import TruthCameraModel
from eczis.rig.schedule import CaptureSchedule
from eczis.schemas import Calibration, CaptureSet
from eczis.targets.base import Target


class SimulatedRig:
    def __init__(self, model: TruthCameraModel, target: Target, calibration: Calibration) -> None:
        self._model = model
        self._target = target
        self._calibration = calibration

    def get_calibration(self) -> Calibration:
        return self._calibration

    def capture_burst(self, schedule: CaptureSchedule, *, root_seed: int) -> CaptureSet:
        schedule.validate()
        frames = [self._model.project(self._target.image, obs) for obs in schedule.observations]
        capture = CaptureSet(
            schema_version="1.0",
            target_id=self._target.target_id,
            root_seed=root_seed,
            calibration=self._calibration,
            frames=frames,
            truth_digest=hashlib.sha256(self._target.image.tobytes()).hexdigest(),
            metadata={"truth_model": self._model.__class__.__name__, "frame_count": len(frames)},
        )
        capture.validate()
        return capture
