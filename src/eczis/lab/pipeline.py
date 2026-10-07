# UNFILED-IP
"""Microscope Lab pipeline: registration -> support field -> question certificate -> next capture.

Assumptions (stated in every result): the calibration profile describes the instrument (blur,
noise, 8-bit quantisation, zoom geometry); each burst's mean position is its reference (zero-mean
vibration); cross-zoom alignment comes from calibration. Without a profile the pipeline refuses.
EXPERIMENTAL: built on eczis.experimental; never changes production M/R/I/U.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.camera.operators import fourier_shift
from eczis.experimental.certificates import (
    Certificate,
    Decision,
    Question,
    best_next_measurement,
    certify,
)
from eczis.experimental.support import Measurement, SupportField, build_operator, support_field
from eczis.lab.commands import Controller, ControlLevel
from eczis.lab.sources import F32, Geometry
from eczis.schemas import ObservationParams

ROI = 32  # reconstruction-grid side of the analysed region (exact posterior is per patch)


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    psf_sigma_lr: float
    read_noise: float
    shot_electrons: float
    quantization_step: float = 1.0 / 255.0
    zoom_levels: tuple[int, ...] = (1,)
    source: str = "unvalidated"  # "simulated" or "measured:<date>"

    def noise_var(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(np.maximum(y, 0.0) / self.shot_electrons + self.read_noise**2 + self.quantization_step**2 / 12.0, dtype=np.float64)


@dataclass
class Burst:
    frames: list[F32]
    zoom: int
    geometry: Geometry


def register(frames: list[F32]) -> list[tuple[float, float]]:
    """Sub-pixel (dx, dy) of each frame in sensor pixels, re-referenced to the burst mean.

    Frames are aliased (that is what super-resolution exploits), so two shifted frames are not
    exact Fourier shifts of each other; matching on low-passed frames removes most of that bias.
    """
    from eczis.camera.operators import gaussian_blur_periodic

    def lowpass(img: np.ndarray) -> np.ndarray:
        return np.asarray(gaussian_blur_periodic(img.astype(np.float32), 1.0), dtype=np.float32)

    ref = lowpass(frames[0])
    shifts = [(0.0, 0.0)]
    grid = np.linspace(-0.6, 0.6, 25)
    for frame in frames[1:]:
        f = lowpass(frame)
        cross = np.fft.ifft2(np.fft.fft2(f) * np.conj(np.fft.fft2(ref))).real
        py, px = np.unravel_index(int(np.argmax(cross)), cross.shape)
        h, w = cross.shape
        iy = py - h if py > h // 2 else py
        ix = px - w if px > w // 2 else px
        best, best_err = (float(ix), float(iy)), np.inf
        for dy in iy + grid:
            for dx in ix + grid:
                err = float(np.sum((fourier_shift(ref, float(dx), float(dy)) - f) ** 2))
                if err < best_err:
                    best, best_err = (float(dx), float(dy)), err
        shifts.append(best)
    mx = float(np.mean([s[0] for s in shifts]))
    my = float(np.mean([s[1] for s in shifts]))
    return [(sx - mx, sy - my) for sx, sy in shifts]


def roi_crop(burst: Burst, roi_hr: tuple[int, int]) -> tuple[list[np.ndarray], int]:
    """Crop each frame to the sensor pixels covering the ROI_HR x ROI_HR region at roi_hr."""
    scale = burst.geometry.scale
    y0 = (roi_hr[0] - burst.geometry.origin_hr[0]) / scale
    x0 = (roi_hr[1] - burst.geometry.origin_hr[1]) / scale
    if not (float(y0).is_integer() and float(x0).is_integer()):
        raise ValueError("ROI must align with the sensor grid at this zoom")
    n = ROI // scale
    yi, xi = int(y0), int(x0)
    h, w = burst.frames[0].shape
    if yi < 0 or xi < 0 or yi + n > h or xi + n > w:
        raise ValueError("ROI is outside this burst's field of view")
    return [f[yi : yi + n, xi : xi + n].astype(np.float64) for f in burst.frames], scale


def burst_measurements(burst: Burst, profile: CalibrationProfile) -> list[Measurement]:
    scale = burst.geometry.scale
    model = EstimatedCameraModel(scale=scale, psf_sigma_lr=profile.psf_sigma_lr)
    shifts = register(burst.frames)
    return [Measurement(model, ObservationParams(f"Z{burst.zoom}F{i}", dx, dy, 0.0, 1.0, 0.0, 0.0, i))
            for i, (dx, dy) in enumerate(shifts)]


@dataclass(frozen=True, slots=True)
class Suggestion:
    action: str
    commands: tuple[str, ...]
    predicted_answer_sd: float
    gain: float
    ranking: tuple[tuple[str, float], ...]


@dataclass
class Analysis:
    certificate: Certificate | None
    support: SupportField | None
    suggestion: Suggestion | None
    status: str
    assumptions: list[str] = field(default_factory=list)
    band_summary: dict[str, str] = field(default_factory=dict)


def presence_question(mask: np.ndarray, support: SupportField) -> Question:
    """'Is the marked feature brighter than its surroundings?' with a data-derived threshold."""
    mean_img = support.mean.reshape(support.shape)
    background = float(np.percentile(mean_img, 20))
    foreground = float(np.percentile(mean_img, 97))
    w = mask.astype(np.float64).ravel()
    return Question("marked_feature_present", w / w.sum(), (background + foreground) / 2.0)


def analyze(
    bursts: list[Burst], profile: CalibrationProfile | None, roi_hr: tuple[int, int], mask: np.ndarray,
    available_zoom: tuple[int, ...] = (1,), burst_size: int = 16,
) -> Analysis:
    assumptions = [
        "calibration profile describes blur, noise, quantisation and zoom geometry",
        "each burst's mean position is its reference (zero-mean vibration)",
        "pixel intensities lie in [0, 1] (prior s.d. 1)",
        "synthetic/prototype: not validated on a physical microscope",
    ]
    if profile is None:
        return Analysis(None, None, None, "UNCALIBRATED: run the calibration wizard first", assumptions)
    rows, ys, vs = [], [], []
    for burst in bursts:
        crops, _ = roi_crop(burst, roi_hr)
        a = build_operator(burst_measurements(burst, profile), (ROI, ROI))
        y = np.concatenate([c.ravel() for c in crops])
        rows.append(a)
        ys.append(y)
        vs.append(profile.noise_var(y))
    a_all, y_all, v_all = np.vstack(rows), np.concatenate(ys), np.concatenate(vs)
    sf = support_field(a_all, y_all, v_all, (ROI, ROI))
    q = presence_question(mask, sf)
    cert = certify(sf, q)
    bands = {}
    for lv in range(1, int(sf.level.max()) + 1):
        cls = sf.coefficient_class[sf.level == lv]
        size = 2**lv
        bands[f"{size}x{size} px detail"] = "supported" if np.all(cls == 2) else ("inferred" if np.all(cls >= 1) else "unsupported")
    suggestion = None
    if cert.decision == Decision.UNDETERMINED:
        suggestion = suggest(sf, q, bursts, profile, available_zoom, burst_size)
    status = {
        Decision.SUPPORTED_ABOVE: "CERTIFIED: feature present",
        Decision.SUPPORTED_BELOW: "CERTIFIED: feature absent",
        Decision.UNDETERMINED: "UNDETERMINED",
    }[cert.decision]
    return Analysis(cert, sf, suggestion, status, assumptions, bands)


def suggest(
    sf: SupportField, q: Question, bursts: list[Burst], profile: CalibrationProfile,
    available_zoom: tuple[int, ...], burst_size: int,
) -> Suggestion:
    """Rank physical actions by predicted answer s.d.; digital zoom is listed with zero gain."""
    current_zoom = max(b.zoom for b in bursts)
    side = int(np.sqrt(burst_size))
    phases = [((i % side) / side, (i // side) / side) for i in range(burst_size)]
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    commands: dict[str, tuple[str, ...]] = {}
    mean = sf.mean
    for zoom in sorted({current_zoom, *available_zoom}):
        if zoom < current_zoom:
            continue
        scale = 4 // zoom
        model = EstimatedCameraModel(scale=scale, psf_sigma_lr=profile.psf_sigma_lr)
        meas = [Measurement(model, ObservationParams(f"P{i}", dx, dy, 0.0, 1.0, 0.0, 0.0, i))
                for i, (dx, dy) in enumerate(phases)]
        a = build_operator(meas, (ROI, ROI))
        name = "repeat burst at current zoom" if zoom == current_zoom else f"optical zoom {zoom}x and capture"
        candidates.append((name, a, profile.noise_var(a @ mean)))
        commands[name] = (f"ZOOM {zoom}", f"CAPTURE {burst_size}") if zoom != current_zoom else (f"CAPTURE {burst_size}",)
    current_sd = float(np.sqrt(q.weights @ sf.cov @ q.weights))
    ranked = best_next_measurement(np.linalg.inv(sf.cov), candidates, q.weights)
    # Digital zoom is appended as a literal at the current s.d.: it is not evaluated by the ranker. The
    # derivation (resampling adds no rows to the operator, so the posterior is unchanged) lives in the
    # support-field certificate matrix; here it is a stated consequence, not a computed one.
    ranking = tuple((name, sd) for name, sd in ranked) + (("digital zoom (adds no measurement)", current_sd),)
    best_name, best_sd = ranked[0]
    return Suggestion(best_name, commands[best_name], best_sd, 1.0 - best_sd / current_sd, ranking)


class CaptureSource(Protocol):
    def geometry(self) -> Geometry: ...

    def capture(self, count: int) -> list[F32]: ...


@dataclass
class InvestigationStep:
    zoom: int
    analysis: Analysis
    executed: tuple[str, ...]


def investigate(
    source: CaptureSource, controller: Controller, profile: CalibrationProfile, roi_hr: tuple[int, int],
    mask: np.ndarray, level: ControlLevel, burst_size: int = 16, max_steps: int = 4, min_gain: float = 0.05,
) -> list[InvestigationStep]:
    """Capture -> analyse -> act until certified, no informative action remains, or steps run out.

    Below AUTOMATIC the suggested commands are returned for the operator instead of executed.
    """
    from eczis.lab.commands import parse

    zoom_of: dict[str, Any] = {"zoom": getattr(source, "zoom", 1)}
    bursts = [Burst(source.capture(burst_size), int(zoom_of["zoom"]), source.geometry())]
    steps: list[InvestigationStep] = []
    for _ in range(max_steps):
        result = analyze(bursts, profile, roi_hr, mask, profile.zoom_levels, burst_size)
        if result.suggestion is None or result.suggestion.gain < min_gain:
            if result.suggestion is not None:
                result.status = "UNDETERMINED: no available action is predicted to resolve this"
            steps.append(InvestigationStep(bursts[-1].zoom, result, ()))
            break
        cmds = result.suggestion.commands
        if level < ControlLevel.AUTOMATIC:
            steps.append(InvestigationStep(bursts[-1].zoom, result, ()))
            break
        for line in cmds:
            reply = controller.send(parse(line))
            if reply.startswith("ERR"):
                raise RuntimeError(reply)
        steps.append(InvestigationStep(bursts[-1].zoom, result, cmds))
        frames = getattr(source, "last_burst", None) or source.capture(burst_size)
        bursts.append(Burst(list(frames), int(getattr(source, "zoom", 1)), source.geometry()))
    return steps
