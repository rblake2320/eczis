# UNFILED-IP
"""Headless Microscope Lab session: the control surface shared by agents (MCP) and scripts.

Holds the active source, controller, calibration profile, marked region and captured bursts.
Every method returns plain JSON-serialisable data. A real camera is opened only by an explicit
open_camera() call. EXPERIMENTAL: built on eczis.experimental; never changes production M/R/I/U.
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from eczis.lab.commands import Controller, ControlLevel, LoggingController, SerialController, parse
from eczis.lab.pipeline import (
    ROI,
    Analysis,
    Burst,
    CalibrationProfile,
    analyze,
    investigate,
)
from eczis.lab.sources import SimulatedMicroscope, glyph_scene

STREAM_FAILED_HINT = ("The camera opened but delivers no frames. Close any other app using it (Windows "
                      "Camera, capture tools, video calls) and retry; on Windows this shows as 0xC00D3704.")


class LabError(RuntimeError):
    """Actionable error for the agent; the message says what to do next."""


def _quick_motion(a: np.ndarray, b: np.ndarray) -> float:
    """Integer-pixel motion via phase correlation on a <=128 px block-averaged copy (cheap).

    The full sub-pixel registration is for analysis only: run per frame on 640x480 video it costs
    seconds and froze the live view (100% CPU).
    """
    step = max(1, int(np.ceil(max(a.shape) / 128)))

    def shrink(x: np.ndarray) -> np.ndarray:
        h, w = (x.shape[0] // step) * step, (x.shape[1] // step) * step
        small = x[:h, :w].astype(np.float64).reshape(h // step, step, w // step, step).mean(axis=(1, 3))
        window = np.outer(np.hanning(small.shape[0]), np.hanning(small.shape[1]))  # tame border edges
        return np.asarray((small - small.mean()) * window)

    fa, fb = np.fft.fft2(shrink(a)), np.fft.fft2(shrink(b))
    r = fb * np.conj(fa)
    r /= np.maximum(np.abs(r), 1e-12)
    corr = np.fft.ifft2(r).real
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    dy = py - h if py > h // 2 else py
    dx = px - w if px > w // 2 else px
    return float(np.hypot(dx, dy) * step)


def frame_quality(frames: list[np.ndarray]) -> dict[str, Any]:
    from scipy.ndimage import laplace

    f = frames[-1]
    focus = float(np.var(laplace(f.astype(np.float32))))
    clipped = float(np.mean((f <= 0.002) | (f >= 0.998)))
    motion = _quick_motion(frames[-2], frames[-1]) if len(frames) >= 2 else 0.0
    return {
        "focus_score": round(focus, 6), "focus": "GOOD" if focus > 2e-4 else "SOFT",
        "clipped_fraction": round(clipped, 4), "exposure": "GOOD" if clipped < 0.01 else "CLIPPING",
        "motion_px": round(motion, 3), "motion": "LOW" if motion < 3.0 else "HIGH",
        "frame_shape": list(f.shape), "mean_level": round(float(f.mean()), 4),
    }


def analysis_dict(result: Analysis) -> dict[str, Any]:
    c, s = result.certificate, result.suggestion
    return {
        "status": result.status,
        "answer_range": None if c is None else [round(c.lower, 4), round(c.upper, 4)],
        "estimate": None if c is None else round(c.estimate, 4),
        "decision": None if c is None else c.decision.value,
        "band_support": result.band_summary,
        "suggested_action": None if s is None else {
            "action": s.action, "commands": list(s.commands), "predicted_gain": round(s.gain, 4),
            "ranking": [{"option": n, "answer_sd": round(sd, 5)} for n, sd in s.ranking],
        },
        "assumptions": result.assumptions,
    }


def _frame_hashes(frames: list[np.ndarray]) -> list[str]:
    import hashlib

    return [hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()[:16] for f in frames]


class LabSession:
    """Control surface. If record_dir is set, every action is appended to <record_dir>/<id>.jsonl."""

    def __init__(self, record_dir: str | Path | None = None) -> None:
        import time
        import uuid

        self.record_dir = Path(record_dir) if record_dir else None
        self.session_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self._reset()

    def _reset(self) -> None:
        self.source: Any = None
        self.controller: Controller = LoggingController()
        self.profile: CalibrationProfile | None = None
        self.roi_hr: tuple[int, int] | None = None
        self.mask: np.ndarray | None = None
        self.bursts: list[Burst] = []
        self.recent: list[np.ndarray] = []
        self.last: dict[str, Any] | None = None
        self.scan_frames: list[np.ndarray] = []
        self.scan_truth: Any = None  # SyntheticScan when simulated
        self.scan_result: Any = None

    @property
    def record_path(self) -> Path | None:
        return None if self.record_dir is None else self.record_dir / f"{self.session_id}.jsonl"

    def record(self, action: str, args: dict[str, Any], result: dict[str, Any], **extra: Any) -> None:
        """Append one capture-record line (no-op without record_dir). Never stores pixel data."""
        if self.record_path is None:
            return
        import time

        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        line = {"t": time.time(), "session": self.session_id, "action": action, "args": args,
                "device": None if self.source is None else self.source.describe(),
                "calibration": None if self.profile is None else self.profile.source,
                "region": None if self.roi_hr is None else list(self.roi_hr), "result": result, **extra}
        with self.record_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")

    # --- sources -------------------------------------------------------------
    def _require_source(self) -> Any:
        if self.source is None:
            raise LabError("No source open. Call eczis_lab_open_simulator or eczis_lab_open_camera first.")
        return self.source

    def close(self) -> dict[str, Any]:
        if self.source is not None and hasattr(self.source, "close"):
            self.source.close()
        self._reset()
        self.record("close", {}, {"closed": True})
        return {"closed": True}

    def open_simulator(self, glyph: str = "8", seed: int = 3) -> dict[str, Any]:
        if glyph not in ("8", "3"):
            raise LabError("glyph must be '8' or '3'")
        self.close()
        sim = SimulatedMicroscope(glyph_scene(glyph == "8"), seed=seed)
        self.source, self.controller = sim, sim
        self.profile = CalibrationProfile(0.70, 0.004, 700.0, zoom_levels=(1, 2, 4), source="simulated")
        out = self.status()
        self.record("open_simulator", {"glyph": glyph, "seed": seed}, out)
        return out

    def open_camera(self, index: int, width: int | None = None, height: int | None = None,
                    fourcc: str = "YUY2", backend: str = "msmf") -> dict[str, Any]:
        import cv2

        from eczis.lab.sources import OpenCVCamera

        self.close()
        api = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}.get(backend)
        if api is None:
            raise LabError("backend must be 'msmf' or 'dshow'")
        try:
            cam = OpenCVCamera(index, api)
        except RuntimeError as exc:
            raise LabError(f"Camera {index} could not be opened via {backend}. Try backend='msmf' "
                           "(default) or another index; list devices with eczis_lab_list_devices.") from exc
        granted = cam.set_mode(width, height, fourcc) if width and height else None
        try:
            cam.wait_for_frame(3.0)
        except RuntimeError as exc:
            cam.close()
            raise LabError(STREAM_FAILED_HINT) from exc
        self.source = cam
        self.controller = LoggingController()  # real hardware: commands go to the operator unless serial
        self.profile = None
        out = self.status()
        out["granted_mode"] = granted
        self.record("open_camera", {"index": index, "width": width, "height": height, "fourcc": fourcc,
                                    "backend": backend}, out)
        return out

    def connect_serial(self, port: str, baudrate: int = 115200) -> dict[str, Any]:
        try:
            self.controller = SerialController(port, baudrate)
        except Exception as exc:  # noqa: BLE001 - pyserial raises several types
            raise LabError(f"Serial port {port} could not be opened: {exc}") from exc
        return {"controller": f"serial {port} @ {baudrate}"}

    # --- state ---------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        src = self.source
        return {
            "source": None if src is None else src.describe(),
            "zoom": None if src is None else getattr(src, "zoom", 1),
            "controller": type(self.controller).__name__,
            "calibration": None if self.profile is None else asdict(self.profile),
            "region": None if self.roi_hr is None or self.mask is None
            else {"roi_hr": list(self.roi_hr), "roi_size": ROI, "mask_pixels": int(self.mask.sum())},
            "bursts": [{"zoom": b.zoom, "frames": len(b.frames)} for b in self.bursts],
            "last_result": None if self.last is None else self.last.get("status"),
        }

    def set_profile(self, psf_sigma_lr: float, read_noise: float, shot_electrons: float,
                    zoom_levels: list[int] | None = None, quantization_step: float = 1.0 / 255.0,
                    source: str = "agent-supplied") -> dict[str, Any]:
        if psf_sigma_lr <= 0 or read_noise <= 0 or shot_electrons <= 0:
            raise LabError("psf_sigma_lr, read_noise and shot_electrons must be positive")
        self.profile = CalibrationProfile(psf_sigma_lr, read_noise, shot_electrons, quantization_step,
                                          tuple(zoom_levels or [1]), source)
        out = {"calibration": asdict(self.profile)}
        self.record("set_profile", out["calibration"], out)
        return out

    def load_profile(self, path: str) -> dict[str, Any]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["zoom_levels"] = list(data.get("zoom_levels", [1]))
        return self.set_profile(**data)

    def set_region(self, roi_y: int, roi_x: int, mask_y0: int, mask_y1: int, mask_x0: int, mask_x1: int) -> dict[str, Any]:
        """ROI corner in reconstruction-grid pixels; mask rectangle relative to the ROI (0..ROI)."""
        if not (0 <= mask_y0 < mask_y1 <= ROI and 0 <= mask_x0 < mask_x1 <= ROI):
            raise LabError(f"mask rectangle must lie within 0..{ROI} and be non-empty")
        mask = np.zeros((ROI, ROI), bool)
        mask[mask_y0:mask_y1, mask_x0:mask_x1] = True
        self.roi_hr, self.mask = (int(roi_y), int(roi_x)), mask
        out = self.status()
        self.record("set_region", {"roi": [roi_y, roi_x], "mask": [mask_y0, mask_y1, mask_x0, mask_x1]}, out["region"])
        return out

    def use_demo_region(self) -> dict[str, Any]:
        return self.set_region(48, 48, 11, 23, 10, 12)

    # --- acquisition ---------------------------------------------------------
    def get_frame(self) -> tuple[dict[str, Any], bytes]:
        src = self._require_source()
        try:
            frame = src.read()
        except RuntimeError as exc:
            raise LabError(STREAM_FAILED_HINT) from exc
        self.recent = (self.recent + [frame])[-2:]
        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8), "L").save(buf, "PNG")
        quality = frame_quality(self.recent)
        self.record("get_frame", {}, quality, frame_hash=_frame_hashes([frame])[0])
        return quality, buf.getvalue()

    def capture_burst(self, count: int = 16) -> dict[str, Any]:
        if not 2 <= count <= 256:
            raise LabError("count must be 2..256")
        src = self._require_source()
        try:
            frames = src.capture(count)
        except RuntimeError as exc:
            raise LabError(STREAM_FAILED_HINT) from exc
        self.bursts.append(Burst(list(frames), int(getattr(src, "zoom", 1)), src.geometry()))
        out = {"captured": count, "zoom": self.bursts[-1].zoom, "bursts_held": len(self.bursts),
               "quality": frame_quality(list(frames[-2:]))}
        self.record("capture_burst", {"count": count}, out, frame_hashes=_frame_hashes(list(frames)))
        return out

    def clear_bursts(self) -> dict[str, Any]:
        self.bursts = []
        return {"bursts_held": 0}

    def send_command(self, line: str) -> dict[str, Any]:
        try:
            cmd = parse(line)
        except ValueError as exc:
            raise LabError(f"{exc}. Valid: FOCUS +n, STAGE_X n, STAGE_Y n, LIGHT <RING|SIDE_LEFT|SIDE_RIGHT|"
                           "TOP|BOTTOM|POLARIZED> 0..100, ZOOM n, CAPTURE n. There is no digital zoom.") from exc
        reply = self.controller.send(cmd)
        if reply.startswith("ERR"):
            raise LabError(f"Device refused {cmd.encode()}: {reply}")
        if cmd.verb == "CAPTURE" and getattr(self.source, "last_burst", None):
            self.bursts.append(Burst(list(self.source.last_burst), int(getattr(self.source, "zoom", 1)),
                                     self.source.geometry()))
        out = {"sent": cmd.encode(), "reply": reply}
        self.record("send_command", {"line": line}, out)
        return out

    # --- analysis ------------------------------------------------------------
    def _require_region(self) -> tuple[tuple[int, int], np.ndarray]:
        if self.roi_hr is None or self.mask is None:
            raise LabError("No region marked. Call eczis_lab_set_region (or eczis_lab_use_demo_region).")
        return self.roi_hr, self.mask

    def analyze(self) -> dict[str, Any]:
        roi, mask = self._require_region()
        if not self.bursts:
            self.capture_burst(16)
        zooms = self.profile.zoom_levels if self.profile else (1,)
        try:
            result = analyze(self.bursts, self.profile, roi, mask, zooms)
        except ValueError as exc:
            raise LabError(f"{exc}. Re-mark the region inside the current field of view.") from exc
        self.last = analysis_dict(result)
        self.record("analyze", {"bursts": [b.zoom for b in self.bursts]}, self.last)
        return self.last

    def investigate(self, level: str = "MANUAL", max_steps: int = 4) -> dict[str, Any]:
        roi, mask = self._require_region()
        src = self._require_source()
        if self.profile is None:
            raise LabError("UNCALIBRATED: set a calibration profile first (eczis_lab_set_profile / load_profile).")
        try:
            lvl = ControlLevel[level.upper()]
        except KeyError as exc:
            raise LabError(f"level must be one of {[x.name for x in ControlLevel]}") from exc
        steps = investigate(src, self.controller, self.profile, roi, mask, lvl, max_steps=max_steps)
        out: dict[str, Any] = {"level": lvl.name, "steps": [{"zoom": s.zoom, "executed": list(s.executed), **analysis_dict(s.analysis)}
                                            for s in steps]}
        self.last = out["steps"][-1]
        self.record("investigate", {"level": lvl.name, "max_steps": max_steps}, out)
        return out


# --- 3D canal scan (EXPERIMENTAL) ------------------------------------------------
def _scan_simulate(self: LabSession, frames: int = 60, twist_deg: float = 45.0, depth_mm: float = 6.0) -> dict[str, Any]:
    from eczis.lab.scan3d import simulate_sweep

    if not 12 <= frames <= 200:
        raise LabError("frames must be 12..200")
    sim = simulate_sweep(frames=frames, twist_deg=twist_deg, depth_mm=depth_mm)
    self.scan_frames, self.scan_truth, self.scan_result = list(sim.frames), sim, None
    out = {"frames": len(sim.frames), "source": "synthetic ear canal (r0 3.5 mm, narrowing + bump)",
           "true_baseline_mm_frame0_to_6": round(float(np.linalg.norm(sim.centres[6] - sim.centres[0])), 4)}
    self.record("scan3d_simulate", {"frames": frames, "twist_deg": twist_deg, "depth_mm": depth_mm}, out)
    return out


def _scan_record(self: LabSession, frames: int = 90, stride: int = 2) -> dict[str, Any]:
    src = self._require_source()
    if not 12 <= frames <= 400 or not 1 <= stride <= 10:
        raise LabError("frames must be 12..400 and stride 1..10")
    try:
        raw = src.capture(frames * stride)
    except RuntimeError as exc:
        raise LabError(STREAM_FAILED_HINT) from exc
    kept = list(raw[::stride])
    self.scan_frames, self.scan_truth, self.scan_result = [np.asarray(f, np.float64) for f in kept], None, None
    out = {"frames": len(self.scan_frames), "frame_shape": list(self.scan_frames[0].shape),
           "hint": "move slowly in and out with a slight twist while recording"}
    self.record("scan3d_record", {"frames": frames, "stride": stride}, out, frame_hashes=_frame_hashes(kept))
    return out


def _scan_reconstruct(self: LabSession, baseline_mm: float | None = None, init_gap: int = 6,
                      fx: float | None = None, fy: float | None = None) -> dict[str, Any]:
    from eczis.lab.scan3d import Intrinsics, reconstruct, summarize

    if len(self.scan_frames) <= init_gap:
        raise LabError("No scan recorded. Call eczis_lab_scan3d_simulate or eczis_lab_scan3d_record first.")
    h, w = self.scan_frames[0].shape
    notes: list[str] = []
    if self.scan_truth is not None:
        k = self.scan_truth.k
        if baseline_mm is None:
            baseline_mm = float(np.linalg.norm(self.scan_truth.centres[init_gap] - self.scan_truth.centres[0]))
    else:
        if fx is None:
            notes.append("UNCALIBRATED lens: focal length guessed as 0.8 x width; geometry is approximate")
        k = Intrinsics(fx or 0.8 * w, fy or fx or 0.8 * w, w / 2.0, h / 2.0, w, h)
        if baseline_mm is None:
            notes.append("NO SCALE: baseline unknown, distances are in arbitrary units")
            baseline_mm = 1.0
    try:
        self.scan_result = reconstruct(self.scan_frames, k, baseline_mm, init_gap=init_gap)
    except Exception as exc:  # noqa: BLE001 - OpenCV raises cv2.error
        raise LabError(f"Reconstruction failed ({exc}). Record a slower sweep with more texture in view.") from exc
    out = summarize(self.scan_result)
    out["notes"] = out["notes"] + notes
    if self.scan_truth is not None:
        sim = self.scan_truth
        r0, c0 = sim.rotations[0], sim.centres[0]
        errs: dict[str, list[float]] = {"R": [], "I": [], "U": []}
        for p in self.scan_result.points:
            wp = r0 @ p.xyz + c0
            tr = sim.canal.radius(np.array([np.arctan2(wp[1], wp[0])]), np.array([wp[2]]))[0]
            errs[p.support].append(abs(float(np.hypot(wp[0], wp[1])) - tr))
        out["synthetic_truth_check_median_radial_error_mm"] = {c: round(float(np.median(v)), 3) for c, v in errs.items() if v}
    self.record("scan3d_reconstruct", {"baseline_mm": baseline_mm, "init_gap": init_gap, "fx": fx}, out)
    return out


def _scan_export(self: LabSession, path: str = "artifacts/scan3d/scan.ply") -> dict[str, Any]:
    from eczis.lab.scan3d import export_ply

    if self.scan_result is None:
        raise LabError("Nothing reconstructed yet. Call eczis_lab_scan3d_reconstruct first.")
    n = export_ply(self.scan_result.points, path)
    out = {"path": str(Path(path).resolve()), "points": n, "colours": "R blue, I amber, U grey"}
    self.record("scan3d_export", {"path": path}, out)
    return out


LabSession.scan3d_simulate = _scan_simulate  # type: ignore[attr-defined]
LabSession.scan3d_record = _scan_record  # type: ignore[attr-defined]
LabSession.scan3d_reconstruct = _scan_reconstruct  # type: ignore[attr-defined]
LabSession.scan3d_export = _scan_export  # type: ignore[attr-defined]


def list_devices() -> dict[str, Any]:
    """Cameras Windows reports, WITHOUT opening any of them."""
    import subprocess

    ps = ("Get-PnpDevice -PresentOnly -Class Camera,Image | Select-Object Status,FriendlyName,InstanceId | "
          "ConvertTo-Json -Compress")
    try:
        proc = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
        data = json.loads(proc.stdout or "[]")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"devices": [], "error": f"device listing unavailable: {exc}"}
    devices = data if isinstance(data, list) else [data]
    return {"devices": [{"name": d.get("FriendlyName"), "status": d.get("Status"), "id": d.get("InstanceId")} for d in devices],
            "note": "Indexes for eczis_lab_open_camera are not guaranteed to follow this order; "
                    "open_camera reports the granted resolution so you can tell devices apart."}
