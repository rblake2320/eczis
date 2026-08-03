# UNFILED-IP
from __future__ import annotations

from dataclasses import dataclass

from eczis.schemas import ObservationParams


@dataclass(frozen=True, slots=True)
class CaptureSchedule:
    observations: tuple[ObservationParams, ...]

    def validate(self) -> None:
        if len(self.observations) < 2:
            raise ValueError("At least two observations are required")
        phases = {(round(o.dx_lr, 6), round(o.dy_lr, 6)) for o in self.observations}
        if len(phases) < min(4, len(self.observations)):
            raise ValueError("Capture schedule lacks distinct subpixel phases")


def lattice_schedule(frame_count: int, root_seed: int) -> CaptureSchedule:
    side = int(frame_count**0.5)
    if side * side != frame_count:
        raise ValueError("Demo schedule requires a square frame count")
    observations: list[ObservationParams] = []
    idx = 0
    for iy in range(side):
        for ix in range(side):
            observations.append(
                ObservationParams(
                    frame_id=f"F{idx:03d}",
                    dx_lr=(ix + 0.17 * (iy % 2)) / side,
                    dy_lr=(iy + 0.11 * (ix % 2)) / side,
                    focus_sigma_delta_lr=0.025 * ((idx % 3) - 1),
                    exposure_scale=1.0 + 0.018 * ((idx % 5) - 2),
                    illumination_x=0.035 * ((idx % 4) - 1.5),
                    illumination_y=0.03 * (((idx // side) % 4) - 1.5),
                    seed=root_seed + 7919 * (idx + 1),
                )
            )
            idx += 1
    schedule = CaptureSchedule(tuple(observations))
    schedule.validate()
    return schedule
