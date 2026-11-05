# UNFILED-IP
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eczis.atomic import atomic_write_text
from eczis.bench.datasets import SPLITS, TARGET_FAMILIES, require_split
from eczis.bench.pipeline import (
    estimated_model_for,
    run_case,
    truth_model_for,
)
from eczis.bench.report import write_benchmark_report, write_report
from eczis.bench.stats import bootstrap_ci
from eczis.config import DemoConfig
from eczis.export.heatmap import provenance_rgba
from eczis.export.payload import write_offline_viewer
from eczis.field.tiled_raster import TiledRasterRenderer
from eczis.schemas import ProvenanceClass
from eczis.seal.bundle import seal_directory, verify_bundle
from eczis.seal.environment import replay_environment
from eczis.seal.keys import development_key
from eczis.seal.ledger import append_event, verify_ledger, write_anchor


def _save_gray(path: Path, array: np.ndarray, *, normalize: bool = False) -> None:
    values = array.astype(np.float32)
    if normalize:
        lo, hi = float(np.percentile(values, 1)), float(np.percentile(values, 99))
        values = (values - lo) / max(hi - lo, 1e-8)
    pixels = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def _save_montage(path: Path, frames: list[np.ndarray]) -> None:
    count = len(frames)
    side = int(np.ceil(np.sqrt(count)))
    h, w = frames[0].shape
    canvas = np.zeros((side * h, side * w), dtype=np.float32)
    for idx, frame in enumerate(frames):
        y, x = divmod(idx, side)
        canvas[y * h : (y + 1) * h, x * w : (x + 1) * w] = frame
    _save_gray(path, canvas)


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"


def run_demo(output: Path, config: DemoConfig | None = None) -> dict[str, Any]:
    cfg = config or DemoConfig()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    ledger = artifacts / "ledger.jsonl"
    append_event(ledger, "run_started", {"root_seed": cfg.root_seed, "config": asdict(cfg)})

    case = run_case("microglyph", cfg)
    capture = case.capture
    truth_model = truth_model_for(cfg)
    estimated_model = estimated_model_for(cfg)
    append_event(
        ledger,
        "capture_completed",
        {"target_id": case.target.target_id, "frame_count": len(capture.frames), "truth_digest": capture.truth_digest},
    )
    append_event(
        ledger,
        "reconstruction_completed",
        {
            "solver_info": case.solver.info,
            "iterations": case.solver.iterations,
            "frame_weights": case.solver.frame_weights,
        },
    )
    detection = float(case.metrics["gate"]["challenge_detection_rate"])
    append_event(
        ledger,
        "challenge_completed",
        {
            "amplitude": case.amplitude,
            "detection_rate": detection,
            "gate_threshold": cfg.gate_z2_threshold,
            "max_frame_threshold": cfg.gate_max_frame_z2_threshold,
        },
    )

    report: dict[str, Any] = {
        "schema_version": "1.1",
        "claim": "Infinitely sampleable output, finite and explicitly classified evidence.",
        "synthetic_only": True,
        "root_seed": cfg.root_seed,
        "models": {
            "truth": truth_model.__class__.__name__,
            "estimated": estimated_model.__class__.__name__,
            "psf_sigma_lr_truth": cfg.psf_sigma_lr_truth,
            "psf_sigma_lr_estimated": cfg.psf_sigma_lr_estimated,
        },
        "methods": case.metrics["methods"],
        "gate": {
            "threshold": cfg.gate_z2_threshold,
            "max_frame_threshold": cfg.gate_max_frame_z2_threshold,
            **case.metrics["gate"],
        },
        "provenance_counts": case.metrics["provenance"]["counts"],
        "provenance": case.metrics["provenance"],
        "solver": asdict(case.solver),
        "environment": replay_environment(),
    }

    _save_gray(artifacts / "truth.png", case.target.image)
    _save_gray(artifacts / "raw_montage.png", np.block([[f.data for f in capture.frames[:4]]]))
    _save_montage(artifacts / "raw_burst.png", [f.data for f in capture.frames])
    _save_gray(artifacts / "bicubic.png", case.bicubic)
    _save_gray(artifacts / "single_frame_deconvolution.png", case.single_frame)
    _save_gray(artifacts / "shift_add.png", case.shift_add)
    _save_gray(artifacts / "map_reconstruction.png", case.map_image)
    _save_gray(artifacts / "cycle_residual.png", case.base_gate.score_hr, normalize=True)
    provenance_rgba(case.provenance, alpha=255).save(artifacts / "provenance.png")
    _save_gray(artifacts / "challenged_candidate.png", case.challenged)
    _save_gray(artifacts / "challenge_residual.png", case.challenge_gate.score_hr, normalize=True)
    provenance_rgba(case.challenge_provenance, alpha=255).save(artifacts / "challenge_provenance.png")
    np.save(artifacts / "truth.npy", case.target.image, allow_pickle=False)
    np.save(artifacts / "map_reconstruction.npy", case.map_image, allow_pickle=False)
    np.save(artifacts / "provenance.npy", case.provenance, allow_pickle=False)
    np.save(artifacts / "confidence.npy", case.confidence, allow_pickle=False)
    np.save(artifacts / "challenge_provenance.npy", case.challenge_provenance, allow_pickle=False)
    np.save(artifacts / "challenge_confidence.npy", case.challenge_confidence, allow_pickle=False)
    atomic_write_text(
        artifacts / "capture.json",
        _json_text(
            {
                "schema_version": capture.schema_version,
                "target_id": capture.target_id,
                "root_seed": capture.root_seed,
                "truth_digest": capture.truth_digest,
                "calibration": asdict(capture.calibration),
                "observations": [asdict(frame.params) for frame in capture.frames],
            }
        ),
    )
    write_report(artifacts, report)

    renderer = TiledRasterRenderer(case.map_image, case.provenance, case.confidence, cfg.scale)
    queries = [
        asdict(renderer.query(0.5, 0.5, 1.0)),
        asdict(renderer.query(0.5, 0.5, float(cfg.scale))),
        asdict(renderer.query(0.5, 0.5, float(cfg.scale * 4))),
    ]
    for item in queries:
        item["provenance"] = ProvenanceClass(item["provenance"]).name
    atomic_write_text(artifacts / "renderer_queries.json", _json_text(queries))

    viewer_metadata: dict[str, Any] = {
        "target_id": case.target.target_id,
        "truth_model": truth_model.__class__.__name__,
        "estimated_model": estimated_model.__class__.__name__,
        "frame_count": len(capture.frames),
        "gate_threshold": cfg.gate_z2_threshold,
        "challenge_detection_rate": detection,
        "evidence_cutoff_cycles_per_hr_px": case.metrics["provenance"]["evidence_cutoff_cycles_per_hr_px"],
        "synthetic_only": True,
    }
    viewer_path = artifacts / "viewer.html"
    write_offline_viewer(
        viewer_path,
        reconstruction=case.map_image,
        provenance=case.provenance,
        confidence=case.confidence,
        residual=case.base_gate.score_hr,
        metadata={**viewer_metadata, "bundle_status": "PENDING SEAL (inside bundle; see verification.json)"},
    )

    ledger_ok, ledger_errors = verify_ledger(ledger)
    if not ledger_ok:
        raise RuntimeError("Ledger failed before sealing: " + "; ".join(ledger_errors))
    append_event(ledger, "preseal_verified", {"ledger_valid": True})
    key = development_key(cfg.root_seed)
    write_anchor(ledger, key)
    key_path = output / "DEMO_ONLY.key"
    key_path.write_bytes(key)
    atomic_write_text(
        output / "KEY_WARNING.txt",
        "This deterministic key is for the synthetic demonstration only. It is not production evidence security.\n",
    )
    bundle = seal_directory(artifacts, key, output / "eczis_demo.eczis")
    verification = verify_bundle(bundle, key)
    if not verification.valid:
        raise RuntimeError("Fresh evidence bundle failed verification: " + "; ".join(verification.errors))
    verification_json = {
        "valid": verification.valid,
        "errors": verification.errors,
        "manifest_digest": verification.manifest_digest,
        "bundle": bundle.name,
    }
    atomic_write_text(output / "verification.json", _json_text(verification_json))
    atomic_write_text(output / "RUN_SUMMARY.json", _json_text(report))
    # The root-level viewer copy is regenerated after verification so its
    # status banner reflects the verified state (the sealed copy cannot change).
    write_offline_viewer(
        output / "ECZIS_VIEWER.html",
        reconstruction=case.map_image,
        provenance=case.provenance,
        confidence=case.confidence,
        residual=case.base_gate.score_hr,
        metadata={
            **viewer_metadata,
            "bundle_status": f"VERIFIED (HMAC dev key) manifest {verification.manifest_digest}",
        },
    )
    shutil.copy2(artifacts / "REPORT.md", output / "REPORT.md")
    return {**report, "output": str(output), "bundle_valid": verification.valid}


SUMMARY_KEYS: tuple[tuple[str, str], ...] = (
    ("methods.map.psnr_db", "MAP PSNR (dB)"),
    ("methods.shift_add.psnr_db", "Shift-add PSNR (dB)"),
    ("methods.single_frame_deconvolution.psnr_db", "Single-frame deconv PSNR (dB)"),
    ("methods.bicubic.psnr_db", "Bicubic PSNR (dB)"),
    ("methods.map.feature_f1", "MAP feature F1"),
    ("provenance.r_fraction", "R fraction"),
    ("provenance.false_r_rate", "False-R rate (supported band)"),
    ("provenance.false_r_rate_full_band", "False-R rate (full band, unobservable incl.)"),
    ("provenance.ece", "ECE (uncalibrated confidence)"),
    ("gate.base_pass_fraction", "Gate pass fraction"),
    ("gate.challenge_detection_rate", "Injected-detail gate demotion rate"),
    ("provenance.injected_pixels_labeled_r_or_m", "Injected pixels labeled R/M"),
)


def _lookup(metrics: dict[str, Any], dotted: str) -> float:
    value: Any = metrics
    for part in dotted.split("."):
        value = value[part]
    return float(value)


def run_benchmark(
    output: Path,
    *,
    split: str = "locked_test",
    families: tuple[str, ...] = TARGET_FAMILIES,
    config: DemoConfig | None = None,
    mismatch_psf_sigma_lr: float | None = 1.0,
) -> dict[str, Any]:
    """Run the family x seed matrix for one registered split.

    Thresholds are fixed constants registered in VALIDATION_PROTOCOL.md; this
    function never tunes on the seeds it evaluates. The mismatch arm re-runs
    the matrix with a wider truth PSF to show transparent degradation.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    unknown = sorted(set(families) - set(TARGET_FAMILIES))
    if unknown:
        raise ValueError(f"unknown target families: {unknown}")
    seeds = SPLITS[split]
    require_split(seeds, (split,), purpose="benchmark")
    base_cfg = config or DemoConfig()
    output.mkdir(parents=True, exist_ok=True)
    arms: dict[str, DemoConfig] = {"nominal": base_cfg}
    if mismatch_psf_sigma_lr is not None:
        arms["psf_mismatch"] = replace(base_cfg, psf_sigma_lr_truth=mismatch_psf_sigma_lr)
    cases: list[dict[str, Any]] = []
    for arm, arm_cfg in arms.items():
        for family in families:
            for seed in seeds:
                metrics = run_case(family, replace(arm_cfg, root_seed=seed)).metrics
                cases.append({"arm": arm, **metrics})
    summary: dict[str, Any] = {}
    for arm in arms:
        summary[arm] = {}
        for family in families:
            rows = [c for c in cases if c["arm"] == arm and c["family"] == family]
            summary[arm][family] = {
                label: bootstrap_ci([_lookup(r, key) for r in rows], seed=0) for key, label in SUMMARY_KEYS
            }
    map_beats_shift_add = sum(
        1
        for family in families
        if summary["nominal"][family]["MAP PSNR (dB)"]["mean"]
        > summary["nominal"][family]["Shift-add PSNR (dB)"]["mean"]
    )
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "claim": "Infinitely sampleable output, finite and explicitly classified evidence.",
        "synthetic_only": True,
        "split": split,
        "seeds": list(seeds),
        "families": list(families),
        "arms": {name: asdict(cfg) for name, cfg in arms.items()},
        "test_set_size": {"cases_per_arm": len(families) * len(seeds), "seeds_per_family": len(seeds)},
        "summary": summary,
        "acceptance": {
            "map_beats_shift_add_families": map_beats_shift_add,
            "family_count": len(families),
            "max_injected_pixels_labeled_r_or_m": max(
                int(c["provenance"]["injected_pixels_labeled_r_or_m"]) for c in cases
            ),
            "max_m_count": max(int(c["provenance"]["m_count"]) for c in cases),
        },
        "cases": cases,
        "environment": replay_environment(),
    }
    atomic_write_text(output / "benchmark_matrix.json", _json_text(result))
    write_benchmark_report(output, result)
    return result
