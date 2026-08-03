# UNFILED-IP
from __future__ import annotations

from pathlib import Path


def test_demo_produces_verified_bundle(demo_run: tuple[Path, dict[str, object]]) -> None:
    output, result = demo_run
    assert result["bundle_valid"] is True
    assert (output / "eczis_demo.eczis").is_file()
    assert (output / "ECZIS_VIEWER.html").is_file()
    assert (output / "REPORT.md").is_file()


def test_map_reconstruction_improves_psnr(demo_run: tuple[Path, dict[str, object]]) -> None:
    _, result = demo_run
    methods = result["methods"]
    assert methods["map"]["psnr_db"] > methods["bicubic"]["psnr_db"]


def test_injected_false_feature_is_demoted(demo_run: tuple[Path, dict[str, object]]) -> None:
    _, result = demo_run
    assert result["gate"]["challenge_detection_rate"] >= 0.95


def test_demo_uses_nonzero_model_mismatch(demo_run: tuple[Path, dict[str, object]]) -> None:
    _, result = demo_run
    models = result["models"]
    assert models["truth"] != models["estimated"]
    assert models["psf_sigma_lr_truth"] != models["psf_sigma_lr_estimated"]
