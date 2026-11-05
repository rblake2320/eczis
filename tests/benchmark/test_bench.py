# UNFILED-IP
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eczis.bench.datasets import (
    CALIBRATION_SEEDS,
    LOCKED_TEST_SEEDS,
    SPLITS,
    SplitLeakageError,
    require_split,
    split_of,
)
from eczis.bench.harness import run_benchmark
from eczis.bench.pipeline import run_case
from eczis.bench.stats import bootstrap_ci
from eczis.config import DemoConfig
from eczis.provenance.calibrate import expected_calibration_error, fit_histogram_calibrator


def test_locked_test_seeds_cannot_be_used_for_calibration() -> None:
    with pytest.raises(SplitLeakageError, match="locked_test"):
        require_split(LOCKED_TEST_SEEDS, ("calibration",), purpose="threshold fitting")
    require_split(CALIBRATION_SEEDS, ("calibration",), purpose="threshold fitting")
    with pytest.raises(SplitLeakageError):
        require_split((999_999,), ("calibration",), purpose="unregistered seed")


def test_splits_are_disjoint() -> None:
    seeds = [s for group in SPLITS.values() for s in group]
    assert len(seeds) == len(set(seeds))
    assert split_of(LOCKED_TEST_SEEDS[0]) == "locked_test"


def test_bootstrap_ci_is_deterministic_and_brackets_mean() -> None:
    values = [1.0, 2.0, 4.0, 8.0]
    first, second = bootstrap_ci(values, seed=0), bootstrap_ci(values, seed=0)
    assert first == second
    assert first["lo"] <= first["mean"] <= first["hi"]
    assert bootstrap_ci([3.0])["lo"] == 3.0
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_calibrator_fitted_on_calibration_improves_held_out_ece(small_cfg: DemoConfig) -> None:
    """ECE target: calibrated confidence on a held-out (development) seed <= 0.10."""
    from dataclasses import replace

    from eczis.bench.pipeline import (
        R_ERROR_TOLERANCE,
        band_limited,
        estimated_model_for,
        local_error,
    )

    def conf_and_correct(seed: int) -> tuple[np.ndarray, np.ndarray]:
        case = run_case("iris", replace(small_cfg, root_seed=seed))
        truth = band_limited(case.target.image, estimated_model_for(small_cfg))
        correct = local_error(truth, case.map_image) <= R_ERROR_TOLERANCE
        return case.confidence.ravel(), correct.ravel().astype(np.float64)

    require_split(CALIBRATION_SEEDS[:1], ("calibration",), purpose="calibrator fit")
    fit_conf, fit_correct = conf_and_correct(CALIBRATION_SEEDS[0])
    calibrator = fit_histogram_calibrator(fit_conf, fit_correct)
    assert list(calibrator.values) == sorted(calibrator.values)  # monotone
    held_conf, held_correct = conf_and_correct(SPLITS["development"][0])
    raw = expected_calibration_error(held_conf, held_correct)
    calibrated = expected_calibration_error(calibrator.apply(held_conf), held_correct)
    assert calibrated <= raw
    assert calibrated <= 0.10


def test_all_five_families_run_and_matrix_reports(small_cfg: DemoConfig, tmp_path: Path) -> None:
    result = run_benchmark(
        tmp_path / "bench", split="development", families=("coin", "pcb"), config=small_cfg,
        mismatch_psf_sigma_lr=1.0,
    )
    assert result["test_set_size"]["cases_per_arm"] == 6
    assert set(result["summary"]) == {"nominal", "psf_mismatch"}
    assert result["acceptance"]["max_injected_pixels_labeled_r_or_m"] == 0
    assert result["acceptance"]["max_m_count"] == 0
    assert (tmp_path / "bench" / "BENCHMARK.md").is_file()
    ci = result["summary"]["nominal"]["coin"]["MAP PSNR (dB)"]
    assert ci["lo"] <= ci["mean"] <= ci["hi"] and ci["n"] == 3.0


def test_mismatch_degrades_transparently(small_cfg: DemoConfig) -> None:
    """A wider truth PSF must not raise false R above the registered limit."""
    from dataclasses import replace

    nominal = run_case("iris", small_cfg).metrics["provenance"]
    mismatch = run_case("iris", replace(small_cfg, psf_sigma_lr_truth=1.0)).metrics["provenance"]
    assert mismatch["false_r_rate"] <= 0.02
    assert nominal["false_r_rate"] <= 0.02


def test_benchmark_rejects_unknown_split_and_family(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_benchmark(tmp_path, split="tuning")
    with pytest.raises(ValueError):
        run_benchmark(tmp_path, families=("moon",))
