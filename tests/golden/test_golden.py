# UNFILED-IP
"""Golden fixtures (T1/T3): freeze serialization exactly and numerics by declared tolerance."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from golden_support import GOLDEN_CFG, fixture_tree  # noqa: E402

from eczis.bench.pipeline import make_target, run_case, simulate_capture  # noqa: E402
from eczis.rig.schedule import lattice_schedule  # noqa: E402
from eczis.seal.canonical import canonical_json  # noqa: E402
from eczis.seal.manifest import build_manifest  # noqa: E402

GOLDEN = json.loads((Path(__file__).parent / "golden_values.json").read_text(encoding="utf-8"))
# Declared cross-platform tolerances (EXECUTION_PLAN Phase 0/8): exact for
# serialization, bounded for floating-point numerics.
TARGET_ATOL = 1e-5
PSNR_ATOL_DB = 0.05
FRACTION_ATOL = 0.01


def test_schedule_serialization_is_frozen() -> None:
    schedule = [asdict(o) for o in lattice_schedule(16, 20260803).observations]
    assert hashlib.sha256(canonical_json(schedule)).hexdigest() == GOLDEN["schedule_sha256"]


def test_manifest_serialization_is_frozen(tmp_path: Path) -> None:
    manifest = build_manifest(fixture_tree(tmp_path), environment={"fixed": True})
    assert hashlib.sha256(canonical_json(manifest)).hexdigest() == GOLDEN["manifest_sha256"]


@pytest.mark.parametrize("family", ["microglyph", "siemens", "coin", "iris", "pcb"])
def test_target_fixtures_within_tolerance(family: str) -> None:
    expected = GOLDEN["targets"][family]
    t = make_target(family, GOLDEN_CFG.hr_shape, GOLDEN_CFG.root_seed)
    h, w = t.image.shape
    blocks = t.image.reshape(4, h // 4, 4, w // 4).mean(axis=(1, 3))
    assert int(t.feature_mask.sum()) == expected["feature_pixels"]
    assert abs(float(t.image.mean()) - expected["mean"]) < TARGET_ATOL
    assert abs(float(t.image.std()) - expected["std"]) < TARGET_ATOL
    assert np.allclose(blocks, np.asarray(expected["block_means"]), atol=TARGET_ATOL)


def test_cross_platform_case_metrics_within_declared_tolerance() -> None:
    expected = GOLDEN["coin_case"]
    m = run_case("coin", GOLDEN_CFG).metrics
    assert abs(m["methods"]["map"]["psnr_db"] - expected["map_psnr_db"]) < PSNR_ATOL_DB
    assert abs(m["methods"]["shift_add"]["psnr_db"] - expected["shift_add_psnr_db"]) < PSNR_ATOL_DB
    assert abs(m["provenance"]["r_fraction"] - expected["r_fraction"]) < FRACTION_ATOL
    assert abs(m["provenance"]["false_r_rate"] - expected["false_r_rate"]) < FRACTION_ATOL
    assert abs(m["gate"]["base_pass_fraction"] - expected["gate_pass_fraction"]) < FRACTION_ATOL


def test_capture_replay_is_bit_identical_same_platform() -> None:
    target = make_target("iris", GOLDEN_CFG.hr_shape, GOLDEN_CFG.root_seed)
    first = simulate_capture(target, GOLDEN_CFG)
    second = simulate_capture(target, GOLDEN_CFG)
    for a, b in zip(first.frames, second.frames, strict=True):
        assert a.data.tobytes() == b.data.tobytes()
        assert np.array_equal(a.valid_mask, b.valid_mask)
