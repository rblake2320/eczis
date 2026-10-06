# UNFILED-IP
"""Observation-invisible false detail (A h ~= 0) must not be labelled Reconstructed.

Pre-registered expectation (2026-10-06, before running): the cycle gate cannot see
h because A(x + h) == A x, and observability is content-independent, so the current
classifier labels such detail R. These tests are strict xfails that document the gap;
when a fix lands they will XPASS and fail the suite until the marker is removed.
"""
from __future__ import annotations

import numpy as np
import pytest

from eczis.bench.scenario import Scenario, high_passed_glyph, null_space_detail, run_scenario
from eczis.schemas import ProvenanceClass


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return run_scenario("microglyph")


def _center(s: Scenario) -> tuple[int, int]:
    h, w = s.reconstruction.shape
    return h // 2, w // 2


def test_null_space_detail_is_invisible_to_observations(scenario: Scenario) -> None:
    detail, _ = null_space_detail(scenario.reconstruction.shape, _center(scenario), amplitude=0.35, radius=8)
    assert float(np.abs(detail).max()) > 0.3
    for frame in scenario.capture.frames:
        assert float(np.abs(scenario.model.project(detail, frame.params)).max()) < 1e-6


@pytest.mark.xfail(strict=True, reason="KNOWN GAP: gate+observability cannot see null-space detail")
def test_invisible_detail_is_not_promoted_to_reconstructed(scenario: Scenario) -> None:
    detail, mask = null_space_detail(scenario.reconstruction.shape, _center(scenario), amplitude=0.35, radius=8)
    _, classes, _ = scenario.evaluate(scenario.reconstruction + detail)
    assert int((classes[mask] >= ProvenanceClass.R).sum()) == 0


@pytest.mark.xfail(strict=True, reason="KNOWN GAP: high-frequency hallucinated texture passes as R")
def test_high_passed_fake_glyph_is_not_promoted_to_reconstructed(scenario: Scenario) -> None:
    detail, mask = high_passed_glyph(
        scenario.reconstruction.shape, _center(scenario), amplitude=0.7, model=scenario.model
    )
    _, classes, _ = scenario.evaluate(scenario.reconstruction + detail)
    assert int((classes[mask] >= ProvenanceClass.R).sum()) == 0
