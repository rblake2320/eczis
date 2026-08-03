# UNFILED-IP
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

from eczis import __version__
from eczis.bench.harness import run_demo
from eczis.camera.calibration import calibration_from_config
from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.camera.truth_model import TruthCameraModel
from eczis.config import DemoConfig
from eczis.export.heatmap import provenance_rgba
from eczis.export.payload import write_offline_viewer
from eczis.gate.cycle import consistency_gate
from eczis.gate.thresholds import GateThresholds
from eczis.provenance.accountant import classify_provenance
from eczis.recon.map_solver import map_reconstruct
from eczis.recon.shift_add import bicubic_baseline, shift_add_reconstruct
from eczis.rig.schedule import lattice_schedule
from eczis.rig.simulated import SimulatedRig
from eczis.seal.bundle import seal_directory, verify_bundle
from eczis.seal.keys import development_key
from eczis.storage import load_capture, save_capture
from eczis.targets.microglyph import create_microglyph_target


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _save_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def doctor(_: argparse.Namespace) -> int:
    checks = {
        "eczis_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "truth_model": TruthCameraModel.__name__,
        "estimated_model": EstimatedCameraModel.__name__,
        "models_are_distinct": TruthCameraModel is not EstimatedCameraModel,
        "claim_language": "Infinitely sampleable output, finite and explicitly classified evidence.",
    }
    _json(checks)
    return 0 if checks["models_are_distinct"] else 2


def demo(args: argparse.Namespace) -> int:
    cfg = replace(
        DemoConfig(),
        root_seed=args.seed,
        frame_count=args.frames,
        max_iterations=args.iterations,
    )
    result = run_demo(args.output, cfg)
    _json(
        {
            "status": "PASS" if result["bundle_valid"] else "FAIL",
            "output": result["output"],
            "bundle_valid": result["bundle_valid"],
            "map_psnr_db": result["methods"]["map"]["psnr_db"],
            "challenge_detection_rate": result["gate"]["challenge_detection_rate"],
            "viewer": str(args.output / "ECZIS_VIEWER.html"),
        }
    )
    return 0


def target_create(args: argparse.Namespace) -> int:
    cfg = DemoConfig(root_seed=args.seed)
    target = create_microglyph_target(cfg.hr_shape, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "target.npy", target.image, allow_pickle=False)
    np.save(args.output / "feature_mask.npy", target.feature_mask, allow_pickle=False)
    _save_gray(args.output / "target.png", target.image)
    (args.output / "target.json").write_text(
        json.dumps({"target_id": target.target_id, "feature_count": target.feature_count, "config": asdict(cfg)}, indent=2),
        encoding="utf-8",
    )
    _json({"status": "PASS", "target_id": target.target_id, "output": args.output})
    return 0


def capture_simulate(args: argparse.Namespace) -> int:
    cfg = DemoConfig(root_seed=args.seed, frame_count=args.frames)
    target_image = np.load(args.target / "target.npy", allow_pickle=False).astype(np.float32)
    feature_mask = np.load(args.target / "feature_mask.npy", allow_pickle=False).astype(np.bool_)
    meta = json.loads((args.target / "target.json").read_text(encoding="utf-8"))
    from eczis.targets.base import Target

    target = Target(meta["target_id"], target_image, feature_mask, int(meta["feature_count"]))
    calibration = calibration_from_config(cfg)
    truth = TruthCameraModel(
        scale=cfg.scale,
        psf_sigma_lr=cfg.psf_sigma_lr_truth,
        read_noise_std=cfg.read_noise_std,
        shot_electrons=cfg.shot_electrons,
        saturation_level=cfg.saturation_level,
    )
    capture = SimulatedRig(truth, target, calibration).capture_burst(
        lattice_schedule(cfg.frame_count, cfg.root_seed), root_seed=cfg.root_seed
    )
    save_capture(capture, args.output)
    _json({"status": "PASS", "frames": len(capture.frames), "output": args.output})
    return 0


def reconstruct(args: argparse.Namespace) -> int:
    capture = load_capture(args.capture)
    model = EstimatedCameraModel(
        scale=capture.calibration.scale,
        psf_sigma_lr=capture.calibration.estimated_psf_sigma_lr,
    )
    bicubic = bicubic_baseline(capture)
    shift_add = shift_add_reconstruct(capture)
    mapped, report = map_reconstruct(
        capture,
        model,
        regularization=args.regularization,
        max_iterations=args.iterations,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, array in (("bicubic", bicubic), ("shift_add", shift_add), ("map", mapped)):
        np.save(args.output / f"{name}.npy", array, allow_pickle=False)
        _save_gray(args.output / f"{name}.png", array)
    (args.output / "solver.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    _json({"status": "PASS", "output": args.output, "solver": asdict(report)})
    return 0


def gate(args: argparse.Namespace) -> int:
    capture = load_capture(args.capture)
    proposal = np.load(args.reconstruction, allow_pickle=False).astype(np.float32)
    model = EstimatedCameraModel(
        scale=capture.calibration.scale,
        psf_sigma_lr=capture.calibration.estimated_psf_sigma_lr,
    )
    result = consistency_gate(
        proposal,
        capture,
        model,
        GateThresholds(args.threshold, min_valid_frames=max(4, len(capture.frames) // 4)),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "score.npy", result.score_hr, allow_pickle=False)
    np.save(args.output / "pass_mask.npy", result.pass_mask_hr, allow_pickle=False)
    (args.output / "gate.json").write_text(json.dumps(result.reasons, indent=2), encoding="utf-8")
    _json({"status": "PASS", "pass_fraction": float(result.pass_mask_hr.mean()), "output": args.output})
    return 0


def provenance(args: argparse.Namespace) -> int:
    capture = load_capture(args.capture)
    score = np.load(args.gate / "score.npy", allow_pickle=False).astype(np.float32)
    passed = np.load(args.gate / "pass_mask.npy", allow_pickle=False).astype(np.bool_)
    from eczis.schemas import GateResult

    result = GateResult(score[:: capture.calibration.scale, :: capture.calibration.scale], score, passed, {})
    classes, confidence = classify_provenance(capture, result)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "provenance.npy", classes, allow_pickle=False)
    np.save(args.output / "confidence.npy", confidence, allow_pickle=False)
    provenance_rgba(classes, alpha=255).save(args.output / "provenance.png")
    _json({"status": "PASS", "output": args.output})
    return 0


def render(args: argparse.Namespace) -> int:
    image = np.load(args.reconstruction, allow_pickle=False).astype(np.float32)
    classes = np.load(args.provenance / "provenance.npy", allow_pickle=False).astype(np.uint8)
    confidence = np.load(args.provenance / "confidence.npy", allow_pickle=False).astype(np.float32)
    residual = np.load(args.gate / "score.npy", allow_pickle=False).astype(np.float32)
    write_offline_viewer(
        args.output,
        reconstruction=image,
        provenance=classes,
        confidence=confidence,
        residual=residual,
        metadata={"synthetic_only": True, "source": "staged CLI workflow"},
    )
    _json({"status": "PASS", "output": args.output})
    return 0


def seal(args: argparse.Namespace) -> int:
    key = development_key(args.seed) if args.key is None else args.key.read_bytes()
    sealed = seal_directory(args.input, key, args.output)
    _json({"status": "PASS", "output": sealed, "development_key": args.key is None})
    return 0


def verify(args: argparse.Namespace) -> int:
    key = development_key(args.seed) if args.key is None else args.key.read_bytes()
    report = verify_bundle(args.bundle, key)
    _json(asdict(report))
    return 0 if report.valid else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eczis", description="ECZIS synthetic software proof")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor")
    p.set_defaults(func=doctor)

    p = sub.add_parser("demo")
    p.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument("--frames", type=int, default=16, choices=(4, 9, 16, 25))
    p.add_argument("--iterations", type=int, default=36)
    p.set_defaults(func=demo)

    target = sub.add_parser("target")
    target_sub = target.add_subparsers(dest="target_command", required=True)
    p = target_sub.add_parser("create")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260803)
    p.set_defaults(func=target_create)

    capture = sub.add_parser("capture")
    capture_sub = capture.add_subparsers(dest="capture_command", required=True)
    p = capture_sub.add_parser("simulate")
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument("--frames", type=int, default=16, choices=(4, 9, 16, 25))
    p.set_defaults(func=capture_simulate)

    p = sub.add_parser("reconstruct")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--regularization", type=float, default=0.0025)
    p.add_argument("--iterations", type=int, default=36)
    p.set_defaults(func=reconstruct)

    p = sub.add_parser("gate")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--reconstruction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=8.0)
    p.set_defaults(func=gate)

    p = sub.add_parser("provenance")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=provenance)

    p = sub.add_parser("render")
    p.add_argument("--reconstruction", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=render)

    p = sub.add_parser("seal")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--key", type=Path)
    p.add_argument("--seed", type=int, default=20260803)
    p.set_defaults(func=seal)

    p = sub.add_parser("verify")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--key", type=Path)
    p.add_argument("--seed", type=int, default=20260803)
    p.set_defaults(func=verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
