# UNFILED-IP
"""ECZIS Lab MCP server (stdio): lets an AI agent drive the microscope/camera loop.

    uv run python -m eczis.lab.mcp_server

Tools are prefixed eczis_lab_. Safety: a real camera opens only via eczis_lab_open_camera;
instrument commands are validated against the protocol; below AUTOMATIC the investigation only
recommends; the server never raises an evidence class and never offers digital zoom as evidence.
EXPERIMENTAL: synthetic reduction-to-practice until validated on calibrated physical hardware.
"""
from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.mcpserver import Image, MCPServer
from mcp.types import ToolAnnotations

from eczis.lab.session import LabError, LabSession, list_devices

INSTRUCTIONS = """Drive the ECZIS evidence-aware imaging loop.
Typical flow: open a source (eczis_lab_open_simulator for practice, eczis_lab_open_camera for
hardware) -> set a calibration profile (simulator sets one; real devices need eczis_lab_set_profile
or eczis_lab_load_profile) -> mark a region (eczis_lab_set_region / eczis_lab_use_demo_region) ->
eczis_lab_analyze -> follow suggested_action, or eczis_lab_investigate with a control level.
Results are CERTIFIED (present/absent), UNDETERMINED (with ranked next captures), or UNCALIBRATED.
Never present an UNDETERMINED result as an answer; digital zoom adds no measurement."""

server: MCPServer[Any] = MCPServer(name="eczis_lab", instructions=INSTRUCTIONS)
session = LabSession(record_dir=os.environ.get("ECZIS_LAB_RECORD_DIR", "artifacts/records"))

READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
ACT = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
HARDWARE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True)


def _call(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        result: dict[str, Any] = fn(*args, **kwargs)
        return result
    except LabError as exc:
        return {"error": str(exc)}


@server.tool(annotations=READ)
def eczis_lab_list_devices() -> dict[str, Any]:
    """List cameras Windows reports (name, status, id) without opening any of them."""
    return list_devices()


@server.tool(annotations=ACT)
def eczis_lab_open_simulator(glyph: str = "8", seed: int = 3) -> dict[str, Any]:
    """Open the simulated microscope (glyph '8' or '3', optical zoom 1/2/4, simulated calibration)."""
    return _call(session.open_simulator, glyph, seed)


@server.tool(annotations=HARDWARE)
def eczis_lab_open_camera(index: int, width: int | None = None, height: int | None = None,
                          fourcc: str = "YUY2", backend: str = "msmf") -> dict[str, Any]:
    """Open a real USB camera/microscope by index (explicit opt-in; it sees whatever it points at).

    Prefer uncompressed fourcc YUY2. Returns the granted mode. Fails with advice if another app
    holds the camera. Real devices start UNCALIBRATED.
    """
    return _call(session.open_camera, index, width, height, fourcc, backend)


@server.tool(annotations=ACT)
def eczis_lab_close() -> dict[str, Any]:
    """Close the current source and reset the session (region, bursts, profile)."""
    return session.close()


@server.tool(annotations=READ)
def eczis_lab_status() -> dict[str, Any]:
    """Current source, zoom, controller, calibration, marked region, held bursts and last result."""
    return session.status()


@server.tool(annotations=READ)
def eczis_lab_get_record(last: int = 20) -> dict[str, Any]:
    """Capture record for this session: path and the last N JSON lines (actions, args, frame hashes, results)."""
    path = session.record_path
    if path is None or not path.exists():
        return {"path": None if path is None else str(path), "entries": []}
    lines = path.read_text(encoding="utf-8").splitlines()
    return {"path": str(path), "total": len(lines), "entries": [json.loads(x) for x in lines[-max(1, last):]]}


@server.tool(annotations=ACT)
def eczis_lab_set_profile(psf_sigma_lr: float, read_noise: float, shot_electrons: float,
                          zoom_levels: list[int] | None = None, quantization_step: float = 1.0 / 255.0,
                          source: str = "agent-supplied") -> dict[str, Any]:
    """Set the calibration profile (blur in sensor px, read noise, shot-noise electrons, zoom levels)."""
    return _call(session.set_profile, psf_sigma_lr, read_noise, shot_electrons, zoom_levels, quantization_step, source)


@server.tool(annotations=ACT)
def eczis_lab_load_profile(path: str) -> dict[str, Any]:
    """Load a calibration profile JSON file."""
    return _call(session.load_profile, path)


@server.tool(annotations=ACT)
def eczis_lab_set_region(roi_y: int, roi_x: int, mask_y0: int, mask_y1: int, mask_x0: int, mask_x1: int) -> dict[str, Any]:
    """Mark the 32x32 analysis region (corner in reconstruction-grid px) and the feature rectangle inside it (0..32)."""
    return _call(session.set_region, roi_y, roi_x, mask_y0, mask_y1, mask_x0, mask_x1)


@server.tool(annotations=ACT)
def eczis_lab_use_demo_region() -> dict[str, Any]:
    """Mark the simulator's demo region: the left stroke that distinguishes '8' from '3'."""
    return _call(session.use_demo_region)


@server.tool(annotations=READ, structured_output=False)
def eczis_lab_get_frame(include_image: bool = False) -> list[Any]:
    """Read one live frame: focus/exposure/motion quality (JSON text), optionally the frame as a PNG image."""
    try:
        quality, png = session.get_frame()
    except LabError as exc:
        return [json.dumps({"error": str(exc)})]
    text = json.dumps(quality)
    return [text, Image(data=png, format="png")] if include_image else [text]


@server.tool(annotations=HARDWARE)
def eczis_lab_capture_burst(count: int = 16) -> dict[str, Any]:
    """Capture a burst of frames at the current optical setting and hold it for analysis."""
    return _call(session.capture_burst, count)


@server.tool(annotations=ACT)
def eczis_lab_clear_bursts() -> dict[str, Any]:
    """Discard held bursts (e.g. after moving the sample)."""
    return session.clear_bursts()


@server.tool(annotations=HARDWARE)
def eczis_lab_send_command(line: str) -> dict[str, Any]:
    """Send one validated instrument command: FOCUS +n, STAGE_X n, STAGE_Y n, LIGHT <channel> 0..100, ZOOM n, CAPTURE n.

    Simulator/serial hardware execute it; otherwise it is queued for the human operator.
    """
    return _call(session.send_command, line)


@server.tool(annotations=HARDWARE)
def eczis_lab_connect_serial(port: str, baudrate: int = 115200) -> dict[str, Any]:
    """Route instrument commands to an Arduino/ESP32 on a serial port (e.g. COM5)."""
    return _call(session.connect_serial, port, baudrate)


@server.tool(annotations=ACT)
def eczis_lab_analyze() -> dict[str, Any]:
    """Analyse held bursts on the marked region: CERTIFIED / UNDETERMINED / UNCALIBRATED, answer range, band support, ranked next captures."""
    return _call(session.analyze)


@server.tool(annotations=HARDWARE)
def eczis_lab_investigate(level: str = "MANUAL", max_steps: int = 4) -> dict[str, Any]:
    """Run capture -> analyse -> act until certified or no action helps.

    level: MANUAL/ASSISTED/SEMI_AUTOMATIC only recommend; AUTOMATIC/AUTONOMOUS execute commands.
    """
    return _call(session.investigate, level, max_steps)


@server.tool(annotations=ACT)
def eczis_lab_scan3d_simulate(frames: int = 60, twist_deg: float = 45.0, depth_mm: float = 6.0) -> dict[str, Any]:
    """Render a synthetic ear-canal sweep with known geometry (for practice and accuracy checks)."""
    return _call(session.scan3d_simulate, frames, twist_deg, depth_mm)  # type: ignore[attr-defined]


@server.tool(annotations=HARDWARE)
def eczis_lab_scan3d_record(frames: int = 90, stride: int = 2) -> dict[str, Any]:
    """Record a 3D sweep from the open camera (keep every stride-th frame). Move slowly in/out with a slight twist."""
    return _call(session.scan3d_record, frames, stride)  # type: ignore[attr-defined]


@server.tool(annotations=ACT)
def eczis_lab_scan3d_reconstruct(baseline_mm: float | None = None, init_gap: int = 6,
                                 fx: float | None = None, fy: float | None = None) -> dict[str, Any]:
    """Reconstruct the recorded sweep: points graded R/I/U, wall coverage and next-motion advice.

    baseline_mm = known distance moved between frame 0 and frame init_gap (gives metric scale).
    fx/fy = calibrated focal length in pixels; omitted on a real camera means UNCALIBRATED.
    """
    return _call(session.scan3d_reconstruct, baseline_mm, init_gap, fx, fy)  # type: ignore[attr-defined]


@server.tool(annotations=ACT)
def eczis_lab_scan3d_export(path: str = "artifacts/scan3d/scan.ply") -> dict[str, Any]:
    """Export the reconstruction as a PLY point cloud coloured by support (opens in Blender/MeshLab)."""
    return _call(session.scan3d_export, path)  # type: ignore[attr-defined]


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
