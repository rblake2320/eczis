# UNFILED-IP
"""The exported render policy is the single source of the zoom-demotion rule."""
from __future__ import annotations

import json
from pathlib import Path

from eczis.field.policy import DEFAULT_POLICY, bilinear_footprint
from eczis.schemas import ProvenanceClass


def test_policy_payload_is_complete_and_ordered() -> None:
    payload = DEFAULT_POLICY.to_payload(4)
    assert payload["order"] == ["U", "I", "R", "M"]
    assert payload["native_scale"] == 4
    assert payload["demote_to_inferred_above"] == 1.5
    assert payload["demote_to_unsupported_above"] == 3.0
    assert "not measured local observability" in payload["basis"]


def test_footprint_weights_sum_to_one_and_drop_zero_weights() -> None:
    for px, py in [(0.0, 0.0), (0.5, 0.0), (0.25, 0.75), (3.0, 3.0), (2.0, 1.5)]:
        fp = bilinear_footprint(px, py, 4, 4)
        assert abs(sum(w for *_, w in fp) - 1.0) < 1e-12
        assert all(w > 0.0 for *_, w in fp)
    assert len(bilinear_footprint(2.0, 2.0, 4, 4)) == 1


def test_scale_cap_is_monotone() -> None:
    caps = [DEFAULT_POLICY.scale_cap(m / 10) for m in range(1, 200)]
    assert all(int(a) >= int(b) for a, b in zip(caps, caps[1:], strict=False))
    assert caps[-1] == ProvenanceClass.U


def test_exported_viewer_carries_policy_and_no_threshold_literals(
    demo_run: tuple[Path, dict[str, object]],
) -> None:
    output, _ = demo_run
    html = (output / "ECZIS_VIEWER.html").read_text(encoding="utf-8")
    payload = json.loads(html.split("const P=", 1)[1].split(";\n", 1)[0])
    assert payload["policy"] == DEFAULT_POLICY.to_payload(4)
    template = (Path(__file__).resolve().parents[2] / "viewer" / "index.html").read_text(encoding="utf-8")
    script = template.split("<script>", 1)[1]
    assert "1.5" not in script and "3.0" not in script
