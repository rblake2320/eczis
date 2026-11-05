# UNFILED-IP
"""Full locked-test matrix (default config). Deselected by default; run with -m full_benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

from eczis.bench.harness import run_benchmark


@pytest.mark.full_benchmark
def test_locked_matrix_acceptance(tmp_path: Path) -> None:
    result = run_benchmark(tmp_path / "bench", split="locked_test")
    acc = result["acceptance"]
    assert acc["map_beats_shift_add_families"] >= 3
    assert acc["max_injected_pixels_labeled_r_or_m"] == 0
    assert acc["max_m_count"] == 0
    for arm in ("nominal", "psf_mismatch"):
        for family in ("coin", "iris"):
            assert result["summary"][arm][family]["False-R rate (supported band)"]["mean"] <= 0.02, (arm, family)
