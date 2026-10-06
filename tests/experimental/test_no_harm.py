# UNFILED-IP
"""No-harm selector: a candidate replaces the baseline only if it predicts held-out frames better.

Pre-registered: a deliberately damaged candidate (MAP + strong fine noise) is never selected; the
selector's decision is recorded with scores for every candidate.
"""
from __future__ import annotations

import numpy as np

from eczis.bench.scenario import run_scenario
from eczis.config import DemoConfig
from eczis.experimental.no_harm import no_harm_select
from eczis.recon.map_solver import map_reconstruct
from eczis.recon.shift_add import bicubic_baseline


def test_damaged_candidate_is_never_selected() -> None:
    s = run_scenario("microglyph", DemoConfig(max_iterations=8))
    rng = np.random.default_rng(0)

    def mapped(c):  # type: ignore[no-untyped-def]
        return map_reconstruct(c, s.model, regularization=s.cfg.regularization, max_iterations=8)[0]

    def damaged(c):  # type: ignore[no-untyped-def]
        return mapped(c) + rng.normal(0, 0.3, s.reconstruction.shape).astype(np.float32)

    decision = no_harm_select(
        {"bicubic": bicubic_baseline, "map": mapped, "damaged": damaged}, "bicubic", s.capture, s.model, [5, 10]
    )
    assert decision.selected != "damaged"
    assert decision.scores["damaged"] > decision.scores["bicubic"]
    assert set(decision.scores) == {"bicubic", "map", "damaged"}
