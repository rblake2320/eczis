# UNFILED-IP
"""Microscope Lab: command protocol, simulated instrument, registration and the investigate loop.

Pre-registered (2026-10-06): at 1x both glyphs are UNDETERMINED; the ranking puts optical zoom
above a repeat burst and digital zoom has zero gain; at AUTONOMOUS level the loop executes the
suggested commands and certifies '8' present and '3' absent with intervals containing the truth;
below AUTOMATIC nothing is executed; hardware without optical zoom ends with "no available action".
"""
from __future__ import annotations

import numpy as np
import pytest

from eczis.lab.commands import Command, ControlLevel, LoggingController, parse
from eczis.lab.pipeline import ROI, Burst, CalibrationProfile, analyze, investigate, register
from eczis.lab.sources import SimulatedMicroscope, glyph_scene

ROI_HR = (48, 48)
PROFILE = CalibrationProfile(0.70, 0.004, 700.0, zoom_levels=(1, 2, 4), source="simulated")


def _mask() -> np.ndarray:
    m = np.zeros((ROI, ROI), bool)
    m[11:23, 10:12] = True
    return m


def _true_answer(scene: np.ndarray) -> float:
    roi = scene[ROI_HR[0] : ROI_HR[0] + ROI, ROI_HR[1] : ROI_HR[1] + ROI]
    return float(roi[_mask()].mean())


@pytest.mark.parametrize("line,encoded", [("FOCUS +12", "FOCUS +12"), ("stage_x -30", "STAGE_X -30"),
                                          ("LIGHT SIDE_LEFT 70", "LIGHT SIDE_LEFT 70"), ("CAPTURE 16", "CAPTURE 16"),
                                          ("ZOOM 4", "ZOOM 4")])
def test_protocol_parses_and_round_trips(line: str, encoded: str) -> None:
    cmd = parse(line)
    assert cmd.encode() == encoded
    assert parse(cmd.encode()) == cmd


@pytest.mark.parametrize("bad", ["FOCUS", "FOCUS 12x", "LIGHT LASER 50", "LIGHT RING 101", "CAPTURE 0",
                                 "CAPTURE 999", "DIGITAL_ZOOM 4", "RM -RF", ""])
def test_protocol_rejects_malformed_and_unknown(bad: str) -> None:
    with pytest.raises(ValueError):
        parse(bad)


def test_logging_controller_and_simulator_replies() -> None:
    log = LoggingController()
    assert log.send(Command("FOCUS", (12,))) == "QUEUED_FOR_OPERATOR" and log.log == ["FOCUS +12"]
    sim = SimulatedMicroscope(glyph_scene(True), zoom_levels=(1, 2))
    assert sim.send(parse("ZOOM 4")).startswith("ERR")
    assert sim.send(parse("ZOOM 2")).startswith("OK") and sim.zoom == 2
    sim.send(parse("CAPTURE 4"))
    assert len(sim.last_burst) == 4 and sim.last_burst[0].dtype == np.float32
    # 8-bit camera output
    assert np.allclose(sim.last_burst[0] * 255, np.round(sim.last_burst[0] * 255), atol=1e-3)


def test_registration_recovers_vibration_within_a_fifth_of_a_pixel() -> None:
    sim = SimulatedMicroscope(glyph_scene(True), seed=1)
    est = np.array(register(sim.capture(16)))
    true = np.array(sim.true_jitter)
    true -= true.mean(axis=0)
    assert np.abs(est - true).max() < 0.2


def test_uncalibrated_analysis_refuses() -> None:
    sim = SimulatedMicroscope(glyph_scene(True))
    burst = Burst(sim.capture(16), 1, sim.geometry())
    result = analyze([burst], None, ROI_HR, _mask())
    assert result.status.startswith("UNCALIBRATED") and result.certificate is None


@pytest.mark.parametrize("eight", [True, False])
def test_autonomous_investigation_certifies_correct_answer(eight: bool) -> None:
    scene = glyph_scene(eight)
    sim = SimulatedMicroscope(scene, seed=3)
    steps = investigate(sim, sim, PROFILE, ROI_HR, _mask(), ControlLevel.AUTONOMOUS)
    first, last = steps[0].analysis, steps[-1].analysis
    assert first.status == "UNDETERMINED"
    assert first.suggestion is not None
    names = [n for n, _ in first.suggestion.ranking]
    assert names.index("optical zoom 4x and capture") < names.index("repeat burst at current zoom")
    digital = dict(first.suggestion.ranking)["digital zoom (adds no measurement)"]
    cert0 = first.certificate
    assert cert0 is not None
    assert digital == pytest.approx((cert0.upper - cert0.lower) / 6.0, rel=1e-6)  # zero gain
    assert steps[0].executed == ("ZOOM 4", "CAPTURE 16")
    assert last.status == ("CERTIFIED: feature present" if eight else "CERTIFIED: feature absent")
    cert = last.certificate
    assert cert is not None
    truth = _true_answer(scene)
    assert cert.lower <= truth <= cert.upper, (truth, cert.lower, cert.upper)


def test_manual_level_only_recommends() -> None:
    sim = SimulatedMicroscope(glyph_scene(True), seed=4)
    steps = investigate(sim, LoggingController(), PROFILE, ROI_HR, _mask(), ControlLevel.MANUAL)
    assert len(steps) == 1 and steps[0].executed == ()
    assert steps[0].analysis.suggestion is not None
    assert steps[0].analysis.suggestion.commands == ("ZOOM 4", "CAPTURE 16")
    assert sim.zoom == 1  # nothing was executed


def test_no_informative_action_without_optical_zoom() -> None:
    sim = SimulatedMicroscope(glyph_scene(False), seed=5, zoom_levels=(1,))
    profile = CalibrationProfile(0.70, 0.004, 700.0, zoom_levels=(1,), source="simulated")
    steps = investigate(sim, sim, profile, ROI_HR, _mask(), ControlLevel.AUTONOMOUS, max_steps=3)
    assert steps[-1].analysis.status.startswith("UNDETERMINED")
    assert all("ZOOM" not in " ".join(s.executed) for s in steps)


@pytest.mark.xfail(strict=True, reason="KNOWN GAP: registration error is not in the uncertainty model; "
                   "intervals undercover (seed 6 for '8': z = -3.49) although every decision is correct")
def test_interval_coverage_includes_registration_uncertainty() -> None:
    misses = 0
    for eight in (True, False):
        for seed in range(3, 9):
            scene = glyph_scene(eight)
            sim = SimulatedMicroscope(scene, seed=seed)
            steps = investigate(sim, sim, PROFILE, ROI_HR, _mask(), ControlLevel.AUTONOMOUS)
            cert = steps[-1].analysis.certificate
            assert cert is not None
            assert cert.answer == ("present" if eight else "absent")  # decisions: always correct
            misses += not (cert.lower <= _true_answer(scene) <= cert.upper)
    assert misses == 0
