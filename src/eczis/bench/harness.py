# UNFILED-IP
from __future__ import annotations

import json
import platform
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eczis.bench.metrics import challenge_detection_rate, feature_scores, psnr, rmse
from eczis.bench.report import write_report
from eczis.camera.calibration import calibration_from_config
from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.camera.truth_model import TruthCameraModel
from eczis.config import DemoConfig
from eczis.export.heatmap import provenance_rgba
from eczis.export.payload import write_offline_viewer
from eczis.field.tiled_raster import TiledRasterRenderer
from eczis.gate.cycle import consistency_gate
from eczis.gate.thresholds import GateThresholds
from eczis.provenance.accountant import classify_provenance
from eczis.recon.map_solver import map_reconstruct
from eczis.recon.shift_add import bicubic_baseline, shift_add_reconstruct
from eczis.rig.schedule import lattice_schedule
from eczis.rig.simulated import SimulatedRig
from eczis.schemas import ProvenanceClass
from eczis.seal.bundle import seal_directory, verify_bundle
from eczis.seal.keys import development_key
from eczis.seal.ledger import append_event, verify_ledger
from eczis.targets.microglyph import create_microglyph_target


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


def _inject_false_feature(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    challenged = image.copy()
    h, w = image.shape
    y0, x0 = h // 2 - 10, w // 2 - 14
    mask = np.zeros_like(image, dtype=np.bool_)
    # A false microglyph-like H whose projected energy is intentionally detectable.
    mask[y0 : y0 + 22, x0 : x0 + 3] = True
    mask[y0 : y0 + 22, x0 + 18 : x0 + 21] = True
    mask[y0 + 9 : y0 + 13, x0 : x0 + 21] = True
    amplitude = 0.72
    challenged[mask] = np.clip(challenged[mask] + amplitude, 0.0, 1.0)
    return challenged.astype(np.float32), mask, amplitude


def _method_metrics(truth: np.ndarray, feature_mask: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    return {
        "psnr_db": psnr(truth, estimate),
        "rmse": rmse(truth, estimate),
        **feature_scores(feature_mask, estimate),
    }


def run_demo(output: Path, config: DemoConfig | None = None) -> dict[str, Any]:
    cfg = config or DemoConfig()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    ledger = artifacts / "ledger.jsonl"
    append_event(ledger, "run_started", {"root_seed": cfg.root_seed, "config": asdict(cfg)})

    target = create_microglyph_target(cfg.hr_shape, seed=cfg.root_seed, stroke_px=1)
    calibration = calibration_from_config(cfg)
    truth_model = TruthCameraModel(
        scale=cfg.scale,
        psf_sigma_lr=cfg.psf_sigma_lr_truth,
        read_noise_std=cfg.read_noise_std,
        shot_electrons=cfg.shot_electrons,
        saturation_level=cfg.saturation_level,
    )
    estimated_model = EstimatedCameraModel(scale=cfg.scale, psf_sigma_lr=cfg.psf_sigma_lr_estimated)
    if truth_model.__class__ is estimated_model.__class__:
        raise RuntimeError("Stop condition: truth and estimated models are identical")
    schedule = lattice_schedule(cfg.frame_count, cfg.root_seed)
    capture = SimulatedRig(truth_model, target, calibration).capture_burst(
        schedule, root_seed=cfg.root_seed
    )
    append_event(
        ledger,
        "capture_completed",
        {"target_id": target.target_id, "frame_count": len(capture.frames), "truth_digest": capture.truth_digest},
    )

    bicubic = bicubic_baseline(capture)
    shift_add = shift_add_reconstruct(capture)
    map_image, solver = map_reconstruct(
        capture,
        estimated_model,
        regularization=cfg.regularization,
        max_iterations=cfg.max_iterations,
    )
    append_event(
        ledger,
        "reconstruction_completed",
        {"solver_info": solver.info, "iterations": solver.iterations, "frame_weights": solver.frame_weights},
    )

    thresholds = GateThresholds(cfg.gate_z2_threshold, min_valid_frames=max(4, cfg.frame_count // 4))
    base_gate = consistency_gate(map_image, capture, estimated_model, thresholds)
    provenance, confidence = classify_provenance(capture, base_gate)

    challenged, injection_mask, amplitude = _inject_false_feature(map_image)
    challenge_gate = consistency_gate(challenged, capture, estimated_model, thresholds)
    challenge_provenance, challenge_confidence = classify_provenance(capture, challenge_gate)
    detection = challenge_detection_rate(injection_mask, base_gate.pass_mask_hr, challenge_gate.pass_mask_hr)
    append_event(
        ledger,
        "challenge_completed",
        {"amplitude": amplitude, "detection_rate": detection, "gate_threshold": cfg.gate_z2_threshold},
    )

    method_metrics = {
        "bicubic": _method_metrics(target.image, target.feature_mask, bicubic),
        "shift_add": _method_metrics(target.image, target.feature_mask, shift_add),
        "map": _method_metrics(target.image, target.feature_mask, map_image),
    }
    counts = {
        ProvenanceClass(index).name: int((provenance == index).sum())
        for index in range(4)
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "claim": "Infinitely sampleable output, finite and explicitly classified evidence.",
        "synthetic_only": True,
        "root_seed": cfg.root_seed,
        "models": {
            "truth": truth_model.__class__.__name__,
            "estimated": estimated_model.__class__.__name__,
            "psf_sigma_lr_truth": cfg.psf_sigma_lr_truth,
            "psf_sigma_lr_estimated": cfg.psf_sigma_lr_estimated,
        },
        "methods": method_metrics,
        "gate": {
            "threshold": cfg.gate_z2_threshold,
            "base_pass_fraction": float(base_gate.pass_mask_hr.mean()),
            "challenge_detection_rate": detection,
            "challenge_amplitude": amplitude,
            "base_reasons": base_gate.reasons,
            "challenge_reasons": challenge_gate.reasons,
        },
        "provenance_counts": counts,
        "solver": asdict(solver),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }

    # Persist evidence artifacts.
    _save_gray(artifacts / "truth.png", target.image)
    _save_gray(artifacts / "raw_montage.png", np.block([[f.data for f in capture.frames[:4]]]))
    _save_montage(artifacts / "raw_burst.png", [f.data for f in capture.frames])
    _save_gray(artifacts / "bicubic.png", bicubic)
    _save_gray(artifacts / "shift_add.png", shift_add)
    _save_gray(artifacts / "map_reconstruction.png", map_image)
    _save_gray(artifacts / "cycle_residual.png", base_gate.score_hr, normalize=True)
    provenance_rgba(provenance, alpha=255).save(artifacts / "provenance.png")
    _save_gray(artifacts / "challenged_candidate.png", challenged)
    _save_gray(artifacts / "challenge_residual.png", challenge_gate.score_hr, normalize=True)
    provenance_rgba(challenge_provenance, alpha=255).save(artifacts / "challenge_provenance.png")
    np.save(artifacts / "truth.npy", target.image, allow_pickle=False)
    np.save(artifacts / "map_reconstruction.npy", map_image, allow_pickle=False)
    np.save(artifacts / "provenance.npy", provenance, allow_pickle=False)
    np.save(artifacts / "confidence.npy", confidence, allow_pickle=False)
    np.save(artifacts / "challenge_provenance.npy", challenge_provenance, allow_pickle=False)
    np.save(artifacts / "challenge_confidence.npy", challenge_confidence, allow_pickle=False)
    (artifacts / "capture.json").write_text(
        json.dumps(
            {
                "schema_version": capture.schema_version,
                "target_id": capture.target_id,
                "root_seed": capture.root_seed,
                "truth_digest": capture.truth_digest,
                "calibration": asdict(capture.calibration),
                "observations": [asdict(frame.params) for frame in capture.frames],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_report(artifacts, report)

    renderer = TiledRasterRenderer(map_image, provenance, confidence, cfg.scale)
    queries = [
        asdict(renderer.query(0.5, 0.5, 1.0)),
        asdict(renderer.query(0.5, 0.5, float(cfg.scale))),
        asdict(renderer.query(0.5, 0.5, float(cfg.scale * 4))),
    ]
    for item in queries:
        item["provenance"] = ProvenanceClass(item["provenance"]).name
    (artifacts / "renderer_queries.json").write_text(
        json.dumps(queries, indent=2, sort_keys=True), encoding="utf-8"
    )

    viewer_path = artifacts / "viewer.html"
    write_offline_viewer(
        viewer_path,
        reconstruction=map_image,
        provenance=provenance,
        confidence=confidence,
        residual=base_gate.score_hr,
        metadata={
            "bundle_status": "PENDING SEAL",
            "target_id": target.target_id,
            "truth_model": truth_model.__class__.__name__,
            "estimated_model": estimated_model.__class__.__name__,
            "frame_count": len(capture.frames),
            "gate_threshold": cfg.gate_z2_threshold,
            "challenge_detection_rate": detection,
            "synthetic_only": True,
        },
    )

    ledger_ok, ledger_errors = verify_ledger(ledger)
    if not ledger_ok:
        raise RuntimeError("Ledger failed before sealing: " + "; ".join(ledger_errors))
    append_event(ledger, "preseal_verified", {"ledger_valid": True})
    key = development_key(cfg.root_seed)
    key_path = output / "DEMO_ONLY.key"
    key_path.write_bytes(key)
    (output / "KEY_WARNING.txt").write_text(
        "This deterministic key is for the synthetic demonstration only. It is not production evidence security.\n",
        encoding="utf-8",
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
    (output / "verification.json").write_text(
        json.dumps(verification_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "RUN_SUMMARY.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copy2(viewer_path, output / "ECZIS_VIEWER.html")
    shutil.copy2(artifacts / "REPORT.md", output / "REPORT.md")
    return {**report, "output": str(output), "bundle_valid": verification.valid}
