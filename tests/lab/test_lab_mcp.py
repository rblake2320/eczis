# UNFILED-IP
"""ECZIS Lab MCP server: session behaviour and a real stdio protocol round trip.

Pre-registered: over stdio an agent can open the simulator, mark the demo region, and an
AUTONOMOUS investigation ends CERTIFIED present for '8'; malformed commands and missing
preconditions return actionable errors instead of crashing; get_frame can return a PNG image.
"""
from __future__ import annotations

import json
import sys
from typing import Any

import anyio
import pytest

pytest.importorskip("mcp")

from eczis.lab.session import LabError, LabSession  # noqa: E402


def test_session_preconditions_are_actionable() -> None:
    s = LabSession()
    with pytest.raises(LabError, match="open_simulator or eczis_lab_open_camera"):
        s.capture_burst(4)
    s.open_simulator("3", seed=4)
    with pytest.raises(LabError, match="No region marked"):
        s.analyze()
    with pytest.raises(LabError, match="no digital zoom"):
        s.send_command("DIGITAL_ZOOM 4")
    with pytest.raises(LabError, match="Device refused"):
        s.send_command("ZOOM 3")
    s.use_demo_region()
    out = s.investigate("AUTONOMOUS")
    assert out["steps"][0]["status"] == "UNDETERMINED"
    assert out["steps"][0]["suggested_action"]["commands"] == ["ZOOM 4", "CAPTURE 16"]
    assert out["steps"][-1]["status"] == "CERTIFIED: feature absent"


def _payload(result: Any) -> Any:
    if getattr(result, "structured_content", None):
        sc = result.structured_content
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    for item in result.content:
        if getattr(item, "type", "") == "text":
            return json.loads(item.text)
    return None


def test_stdio_round_trip_agent_certifies_glyph() -> None:
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "eczis.lab.mcp_server"])

    async def run() -> dict[str, Any]:
        async with Client(params) as client:
            listed = (await client.list_tools()).tools
            tools = {t.name for t in listed}
            got: dict[str, Any] = {"tools": tools, "hints": {t.name: t.annotations for t in listed}}
            got["no_source"] = _payload(await client.call_tool("eczis_lab_capture_burst", {"count": 4}))
            await client.call_tool("eczis_lab_open_simulator", {"glyph": "8", "seed": 3})
            await client.call_tool("eczis_lab_use_demo_region", {})
            got["bad_cmd"] = _payload(await client.call_tool("eczis_lab_send_command", {"line": "RM -RF"}))
            frame = await client.call_tool("eczis_lab_get_frame", {"include_image": True})
            got["frame_types"] = [c.type for c in frame.content]
            got["investigate"] = _payload(await client.call_tool("eczis_lab_investigate", {"level": "AUTONOMOUS"}))
            got["status"] = _payload(await client.call_tool("eczis_lab_status", {}))
            return got

    got = anyio.run(run)
    assert {"eczis_lab_open_camera", "eczis_lab_investigate", "eczis_lab_analyze", "eczis_lab_get_frame"} <= got["tools"]
    assert got["hints"]["eczis_lab_status"].read_only_hint is True  # safety hints reach the client
    assert got["hints"]["eczis_lab_open_camera"].open_world_hint is True
    assert got["hints"]["eczis_lab_investigate"].read_only_hint is False
    assert "No source open" in got["no_source"]["error"]
    assert "unknown command" in got["bad_cmd"]["error"]
    assert "image" in got["frame_types"]
    steps = got["investigate"]["steps"]
    assert steps[0]["status"] == "UNDETERMINED" and steps[0]["executed"] == ["ZOOM 4", "CAPTURE 16"]
    assert steps[-1]["status"] == "CERTIFIED: feature present"
    assert got["status"]["zoom"] == 4


def test_capture_record_logs_every_action_without_pixels(tmp_path) -> None:  # type: ignore[no-untyped-def]
    s = LabSession(record_dir=tmp_path)
    s.open_simulator("8", seed=3)
    s.use_demo_region()
    s.capture_burst(4)
    s.send_command("LIGHT RING 50")
    s.analyze()
    lines = [json.loads(x) for x in s.record_path.read_text(encoding="utf-8").splitlines()]  # type: ignore[union-attr]
    actions = [x["action"] for x in lines]
    assert actions[:2] == ["close", "open_simulator"] and {"set_region", "capture_burst", "send_command", "analyze"} <= set(actions)
    burst = next(x for x in lines if x["action"] == "capture_burst")
    assert len(burst["frame_hashes"]) == 4 and all(len(h) == 16 for h in burst["frame_hashes"])
    assert all(x["session"] == s.session_id for x in lines)
    assert "data" not in json.dumps(lines)  # no pixel payloads in the record


def test_scan3d_preconditions_are_actionable() -> None:
    s = LabSession()
    with pytest.raises(LabError, match="No scan recorded"):
        s.scan3d_reconstruct()  # type: ignore[attr-defined]
    with pytest.raises(LabError, match="Nothing reconstructed"):
        s.scan3d_export()  # type: ignore[attr-defined]
    with pytest.raises(LabError, match="No source open"):
        s.scan3d_record(30)  # type: ignore[attr-defined]
