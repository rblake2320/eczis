# UNFILED-IP
"""Run user-supplied images through ECZIS as synthetic ground truth.

Each image becomes the scene; a simulated burst is captured from it, reconstructed, gated and
classified, then attacked. Two capture modes:

  A  same grid    truth on the reconstruction (HR) grid, TruthCameraModel scale 4
  B  finer grid   truth on a 2x finer grid, TruthCameraModel scale 8 (independent sampling
                  grid -> guards against the "inverse crime"); metrics vs 2x2-averaged truth

Images are third-party/local test inputs: they live under gitignored artifacts/ and are never
committed. Synthetic software reduction-to-practice only.
Usage: uv run python research/real_texture_test.py IMG [IMG ...] [--realesrgan EXE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery import detectable_glyph, rates  # noqa: E402

from eczis.bench.scenario import Scenario, high_passed_glyph, null_space_detail  # noqa: E402
from eczis.camera.calibration import calibration_from_config  # noqa: E402
from eczis.camera.estimated_model import EstimatedCameraModel  # noqa: E402
from eczis.camera.operators import block_average  # noqa: E402
from eczis.camera.truth_model import TruthCameraModel  # noqa: E402
from eczis.config import DemoConfig  # noqa: E402
from eczis.gate.cycle import consistency_gate  # noqa: E402
from eczis.gate.thresholds import GateThresholds  # noqa: E402
from eczis.provenance.accountant import classify_provenance  # noqa: E402
from eczis.recon.map_solver import map_reconstruct  # noqa: E402
from eczis.rig.schedule import lattice_schedule  # noqa: E402
from eczis.rig.simulated import SimulatedRig  # noqa: E402
from eczis.targets.base import Target  # noqa: E402


def load_gray(path: Path, shape: tuple[int, int], crop: tuple[int, int, int, int] | None) -> np.ndarray:
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
    im = Image.alpha_composite(bg, im).convert("L")
    if crop:
        im = im.crop(crop)
    h, w = shape
    sw, sh = im.size
    target_ratio = w / h
    if sw / sh > target_ratio:
        nw = int(sh * target_ratio)
        im = im.crop(((sw - nw) // 2, 0, (sw - nw) // 2 + nw, sh))
    else:
        nh = int(sw / target_ratio)
        im = im.crop((0, (sh - nh) // 2, sw, (sh - nh) // 2 + nh))
    im = im.resize((w, h), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return (0.05 + 0.85 * arr).astype(np.float32)


def edge_mask(img: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(img)
    mag = np.hypot(gx, gy)
    return np.asarray(mag >= np.percentile(mag, 85), dtype=np.bool_)


def build(img_fine: np.ndarray, mode: str, cfg: DemoConfig) -> Scenario:
    if mode == "A":
        truth_img = block_average(img_fine, 2)
        scene, truth_scale = truth_img, cfg.scale
    else:
        truth_img = block_average(img_fine, 2)
        scene, truth_scale = img_fine, cfg.scale * 2
    target = Target(f"user-{mode}", truth_img, edge_mask(truth_img), int(edge_mask(truth_img).sum()))
    scene_target = Target(target.target_id, scene, target.feature_mask, target.feature_count)
    truth = TruthCameraModel(
        scale=truth_scale, psf_sigma_lr=cfg.psf_sigma_lr_truth, read_noise_std=cfg.read_noise_std,
        shot_electrons=cfg.shot_electrons, saturation_level=cfg.saturation_level,
    )
    capture = SimulatedRig(truth, scene_target, calibration_from_config(cfg)).capture_burst(
        lattice_schedule(cfg.frame_count, cfg.root_seed), root_seed=cfg.root_seed
    )
    model = EstimatedCameraModel(scale=cfg.scale, psf_sigma_lr=cfg.psf_sigma_lr_estimated)
    recon, _ = map_reconstruct(capture, model, regularization=cfg.regularization, max_iterations=cfg.max_iterations)
    thresholds = GateThresholds(cfg.gate_z2_threshold, min_valid_frames=max(4, cfg.frame_count // 4))
    gate = consistency_gate(recon, capture, model, thresholds)
    classes, conf = classify_provenance(capture, gate)
    return Scenario(cfg, target, capture, model, thresholds, recon, gate, classes, conf)


def band_limited_accuracy(s: Scenario) -> dict[str, float]:
    from eczis.camera.operators import gaussian_blur_periodic

    out = {}
    for sig in (0, 2):
        t = gaussian_blur_periodic(s.target.image, sig) if sig else s.target.image
        out[f"within_0.05_sigma{sig}"] = float((np.abs(s.reconstruction - t) < 0.05).mean())
    return out


def attacks(s: Scenario) -> dict[str, Any]:
    shape = s.reconstruction.shape
    c = (shape[0] // 2, shape[1] // 2)
    res = {}
    for name, (d, m) in {
        "detectable_glyph_0.72": detectable_glyph(shape, 0.72),
        "null_checker_0.35": null_space_detail(shape, c, amplitude=0.35, radius=8),
        "highpass_glyph_0.7": high_passed_glyph(shape, c, amplitude=0.7, model=s.model),
    }.items():
        p = np.clip(s.reconstruction + d, -1.0, 2.0).astype(np.float32)
        _, classes, _ = s.evaluate(p)
        res[name] = round(float((classes[m] < 2).mean()), 3)  # caught fraction
    return res


def realesrgan(s: Scenario, exe: Path, workdir: Path) -> dict[str, Any]:
    import subprocess

    frame = s.capture.frames[0]
    lr = np.clip(frame.data / np.float32(frame.params.exposure_scale), 0, 1)
    src, dst = (workdir / "lr.png").resolve(), (workdir / "sr.png").resolve()
    Image.fromarray((lr * 255).astype(np.uint8), "L").convert("RGB").save(src)
    proc = subprocess.run([str(exe), "-i", str(src), "-o", str(dst), "-n", "realesrgan-x4plus", "-s", "4"],
                          capture_output=True, text=True, cwd=exe.parent)
    if proc.returncode != 0 or not dst.exists():
        return {"status": "failed"}
    sr = np.asarray(Image.open(dst).convert("L"), dtype=np.float32) / 255.0
    bic = np.asarray(Image.fromarray((lr * 255).astype(np.uint8), "L").resize(sr.shape[::-1], Image.Resampling.BICUBIC),
                     dtype=np.float32) / 255.0
    out: dict[str, Any] = {}
    for label, prop in {"realesrgan": sr, "bicubic": bic}.items():
        gate, classes, _ = s.evaluate(prop.astype(np.float32))
        out[label] = {"R_fraction": round(float((classes >= 2).mean()), 4),
                      "rmse": round(float(np.sqrt(np.mean((prop - s.target.image) ** 2))), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--crop", action="append", default=[], help="name=x0,y0,x1,y1 crop for an image stem")
    ap.add_argument("--realesrgan", type=Path)
    ap.add_argument("--output", type=Path, default=Path("artifacts/real_texture"))
    args = ap.parse_args()
    crops = {k: tuple(int(v) for v in c.split(",")) for k, c in (x.split("=", 1) for x in args.crop)}
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = DemoConfig(max_iterations=24)
    fine_shape = (cfg.hr_shape[0] * 2, cfg.hr_shape[1] * 2)
    report: dict[str, Any] = {"synthetic_only": True, "note": "user images used as simulated ground truth"}
    for path in args.images:
        jobs = [(path.stem, None)] + [(f"{path.stem}-crop", crops[path.stem])] if path.stem in crops else [(path.stem, None)]
        for name, crop in jobs:
            img = load_gray(path, fine_shape, crop)  # type: ignore[arg-type]
            for mode in ("A", "B"):
                s = build(img, mode, cfg)
                row: dict[str, Any] = {**rates(s, s.reconstruction, s.provenance, None), **band_limited_accuracy(s),
                                       "caught": attacks(s)}
                if args.realesrgan and mode == "B":
                    row["realesrgan"] = realesrgan(s, args.realesrgan, args.output)
                report[f"{name}/{mode}"] = row
                Image.fromarray((np.clip(s.reconstruction, 0, 1) * 255).astype(np.uint8), "L").save(
                    args.output / f"{name}_{mode}_recon.png")
                Image.fromarray((np.clip(s.target.image, 0, 1) * 255).astype(np.uint8), "L").save(
                    args.output / f"{name}_truth.png")
                print(f"{name}/{mode} done", flush=True)
    (args.output / "real_texture.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output / 'real_texture.json'}")


if __name__ == "__main__":
    main()
