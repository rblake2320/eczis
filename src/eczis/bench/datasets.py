# UNFILED-IP
"""Dataset split registry and leakage guard.

Thresholds may be fitted on CALIBRATION seeds only; implementation tuning
uses DEVELOPMENT seeds; LOCKED_TEST seeds are for final reporting only.
``require_split`` makes leakage a hard error instead of a convention.
"""

from __future__ import annotations

from typing import Literal

Split = Literal["calibration", "development", "locked_test", "mismatch", "adversarial"]

CALIBRATION_SEEDS = (101, 211, 307)
DEVELOPMENT_SEEDS = (401, 503, 601)
LOCKED_TEST_SEEDS = (701, 809, 907)
MISMATCH_SEEDS = (1009, 1103, 1201)
ADVERSARIAL_SEEDS = (1301, 1409, 1511)

SPLITS: dict[str, tuple[int, ...]] = {
    "calibration": CALIBRATION_SEEDS,
    "development": DEVELOPMENT_SEEDS,
    "locked_test": LOCKED_TEST_SEEDS,
    "mismatch": MISMATCH_SEEDS,
    "adversarial": ADVERSARIAL_SEEDS,
}

TARGET_FAMILIES = ("microglyph", "siemens", "coin", "iris", "pcb")


class SplitLeakageError(RuntimeError):
    """A seed was used for a purpose its split does not allow."""


def split_of(seed: int) -> str | None:
    for name, seeds in SPLITS.items():
        if seed in seeds:
            return name
    return None


def require_split(seeds: tuple[int, ...] | list[int], allowed: tuple[str, ...], *, purpose: str) -> None:
    if not seeds:
        raise SplitLeakageError(f"{purpose}: no seeds supplied")
    for seed in seeds:
        split = split_of(seed)
        if split not in allowed:
            raise SplitLeakageError(
                f"{purpose}: seed {seed} belongs to split {split!r}; allowed splits are {allowed}"
            )


def _check_disjoint() -> None:
    seen: dict[int, str] = {}
    for name, seeds in SPLITS.items():
        for seed in seeds:
            if seed in seen:
                raise SplitLeakageError(f"seed {seed} is in both {seen[seed]} and {name}")
            seen[seed] = name


_check_disjoint()
