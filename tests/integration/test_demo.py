# UNFILED-IP
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eczis.seal.keys import development_key
from eczis.seal.ledger import verify_ledger


def _methods(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    methods: dict[str, dict[str, float]] = result["methods"]
    return methods


def test_demo_produces_verified_bundle(demo_run: tuple[Path, dict[str, Any]]) -> None:
    output, result = demo_run
    assert result["bundle_valid"] is True
    assert (output / "eczis_demo.eczis").is_file()
    assert (output / "ECZIS_VIEWER.html").is_file()
    assert (output / "REPORT.md").is_file()


def test_map_reconstruction_improves_psnr(demo_run: tuple[Path, dict[str, Any]]) -> None:
    _, result = demo_run
    methods = _methods(result)
    assert methods["map"]["psnr_db"] > methods["bicubic"]["psnr_db"]
    assert methods["map"]["psnr_db"] > methods["single_frame_deconvolution"]["psnr_db"]


def test_injected_false_feature_is_demoted(demo_run: tuple[Path, dict[str, Any]]) -> None:
    _, result = demo_run
    assert result["gate"]["challenge_detection_rate"] >= 0.95
    assert result["provenance"]["injected_pixels_labeled_r_or_m"] == 0


def test_demo_uses_nonzero_model_mismatch(demo_run: tuple[Path, dict[str, Any]]) -> None:
    _, result = demo_run
    models = result["models"]
    assert models["truth"] != models["estimated"]
    assert models["psf_sigma_lr_truth"] != models["psf_sigma_lr_estimated"]


def test_demo_ledger_is_anchored_and_sealed(demo_run: tuple[Path, dict[str, Any]]) -> None:
    output, result = demo_run
    ledger = output / "artifacts" / "ledger.jsonl"
    anchor = output / "artifacts" / "ledger.jsonl.anchor.json"
    ok, errors = verify_ledger(ledger, anchor_path=anchor, key=development_key(int(result["root_seed"])))
    assert ok, errors
    manifest = json.loads((output / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
    sealed = {item["path"] for item in manifest["files"]}
    assert {"ledger.jsonl", "ledger.jsonl.anchor.json"} <= sealed
    assert "ledger.jsonl.lock" not in sealed


def test_demo_records_replay_environment(demo_run: tuple[Path, dict[str, Any]]) -> None:
    output, _ = demo_run
    manifest = json.loads((output / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
    env = manifest["environment"]
    for key in ("python", "os", "machine", "numpy", "scipy", "eczis_version", "uv_lock_sha256"):
        assert key in env


def test_root_viewer_shows_verified_status(demo_run: tuple[Path, dict[str, Any]]) -> None:
    output, _ = demo_run
    html = (output / "ECZIS_VIEWER.html").read_text(encoding="utf-8")
    assert "VERIFIED" in html
    assert "http://" not in html and "https://" not in html  # offline: no network fetches


def test_demo_never_emits_m(demo_run: tuple[Path, dict[str, Any]]) -> None:
    _, result = demo_run
    assert result["provenance_counts"]["M"] == 0
