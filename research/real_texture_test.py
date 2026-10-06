# UNFILED-IP
"""Run external texture images through ECZIS as declared synthetic scenes.

User-supplied images are treated solely as synthetic scene textures. All provenance results
refer to newly simulated observations generated from those textures, not to the authenticity or
measurement support of the source files. Results characterize ECZIS's current gate under the
declared conditions; they are not a benchmark of Real-ESRGAN generally.

Capture modes (forward-model dependence):
  A  same grid    truth on the reconstruction (HR) grid, TruthCameraModel scale 4
  B  finer grid   truth on a 2x finer grid, TruthCameraModel scale 8
  C  independent  truth on a 4x finer grid; separate numpy/scipy forward model that never calls
                  ECZIS camera operators: edge-padded integer fine-grid shifts, disk (defocus) x
                  Gaussian PSF, vignetting instead of the linear illumination ramp, pixel-area
                  integration, Poisson + read noise, 12-bit ADC quantization, clipping.
Mode C reduces but does not eliminate inverse-crime risk (same reconstruction family, no lens
distortion or Bayer mosaic).

Proposal attacks are reported separately from capture modes. Inputs and outputs stay in
gitignored artifacts/; missing inputs or tools are reported as NOT RUN / ERROR, never omitted.
Usage: uv run python research/real_texture_test.py IMG [IMG ...] [--crop stem=x0,y0,x1,y1]
       [--realesrgan EXE] [--output DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery import detectable_glyph, rates  # noqa: E402

from eczis.bench.scenario import Scenario, high_passed_glyph, null_space_detail  # noqa: E402
from eczis.camera.calibration import calibration_from_config  # noqa: E402
from eczis.camera.estimated_model import EstimatedCameraModel  # noqa: E402
from eczis.camera.operators import gaussian_blur_periodic  # noqa: E402
from eczis.camera.truth_model import TruthCameraModel  # noqa: E402
from eczis.config import DemoConfig  # noqa: E402
from eczis.export.heatmap import provenance_rgba  # noqa: E402
from eczis.gate.cycle import consistency_gate  # noqa: E402
from eczis.gate.thresholds import GateThresholds  # noqa: E402
from eczis.provenance.accountant import classify_provenance  # noqa: E402
from eczis.recon.map_solver import map_reconstruct  # noqa: E402
from eczis.rig.schedule import lattice_schedule  # noqa: E402
from eczis.rig.simulated import SimulatedRig  # noqa: E402
from eczis.schemas import CaptureSet, RawFrame  # noqa: E402
from eczis.targets.base import Target  # noqa: E402

EPSILONS = (0.02, 0.05, 0.10, 0.15, 0.20)
FINE = 4  # mode C truth grid factor relative to the reconstruction grid


def area_mean(img: np.ndarray, k: int) -> np.ndarray:
    """Pixel-area integration, implemented here (not eczis.camera.operators)."""
    h, w = img.shape
    return np.asarray(img.reshape(h // k, k, w // k, k).mean(axis=(1, 3)), dtype=np.float32)


def load_gray(path: Path, shape: tuple[int, int], crop: tuple[int, int, int, int] | None) -> np.ndarray:
    im = Image.open(path).convert("RGBA")
    im = Image.alpha_composite(Image.new("RGBA", im.size, (0, 0, 0, 255)), im).convert("L")
    if crop:
        im = im.crop(crop)
    h, w = shape
    sw, sh = im.size
    if sw / sh > w / h:
        nw = int(sh * w / h)
        im = im.crop(((sw - nw) // 2, 0, (sw - nw) // 2 + nw, sh))
    else:
        nh = int(sw * h / w)
        im = im.crop((0, (sh - nh) // 2, sw, (sh - nh) // 2 + nh))
    arr = np.asarray(im.resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    return (0.05 + 0.85 * arr).astype(np.float32)


def edge_mask(img: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(img)
    mag = np.hypot(gx, gy)
    return np.asarray(mag >= np.percentile(mag, 85), dtype=np.bool_)


def independent_capture(fine: np.ndarray, cfg: DemoConfig) -> CaptureSet:
    """Mode C forward model: no ECZIS camera operators are used here."""
    lr_k = cfg.scale * FINE  # fine pixels per sensor pixel
    r = 1.2 * lr_k  # defocus disk radius (1.2 sensor px)
    yy, xx = np.mgrid[-int(r) - 1 : int(r) + 2, -int(r) - 1 : int(r) + 2]
    disk = (xx**2 + yy**2 <= r**2).astype(np.float64)
    disk /= disk.sum()
    sg = 0.4 * lr_k
    g = np.arange(-int(3 * sg), int(3 * sg) + 1)
    g1 = np.exp(-(g**2) / (2 * sg**2))
    g1 /= g1.sum()
    psf = fftconvolve(disk, np.outer(g1, g1), mode="full")
    h, w = fine.shape
    vy = np.linspace(-1, 1, h)[:, None]
    vx = np.linspace(-1, 1, w)[None, :]
    vignette = (1.0 - 0.03 * (vx**2 + vy**2)).astype(np.float64)
    pad = int(lr_k * 2)
    frames: list[RawFrame] = []
    for obs in lattice_schedule(cfg.frame_count, cfg.root_seed).observations:
        rng = np.random.default_rng(obs.seed ^ 0x5A5A)
        sx, sy = int(round(obs.dx_lr * lr_k)), int(round(obs.dy_lr * lr_k))
        padded = np.pad(fine.astype(np.float64), pad, mode="edge")
        shifted = padded[pad - sy : pad - sy + h, pad - sx : pad - sx + w]
        scene = shifted * vignette * obs.exposure_scale
        blurred = fftconvolve(np.pad(scene, psf.shape[0], mode="symmetric"), psf, mode="same")
        blurred = blurred[psf.shape[0] : psf.shape[0] + h, psf.shape[1] : psf.shape[1] + w]
        sensor = np.clip(area_mean(blurred.astype(np.float32), lr_k), 0.0, 1.0)
        shot = rng.poisson(sensor * cfg.shot_electrons) / cfg.shot_electrons
        noisy = np.clip(shot + rng.normal(0.0, cfg.read_noise_std, sensor.shape), 0.0, 1.0)
        adc = (np.round(noisy * 4095.0) / 4095.0).astype(np.float32)
        sat = adc >= 0.999
        frames.append(RawFrame(adc, ~sat, sat, obs))
    cap = CaptureSet("1.0", "user-C", cfg.root_seed, calibration_from_config(cfg), frames,
                     hashlib.sha256(fine.tobytes()).hexdigest(), {"truth_model": "independent-modeC"})
    cap.validate()
    return cap


def build(img_fine4: np.ndarray, mode: str, cfg: DemoConfig) -> Scenario:
    truth_img = area_mean(img_fine4, FINE)  # reference on the reconstruction grid
    mask = edge_mask(truth_img)
    target = Target(f"user-{mode}", truth_img, mask, int(mask.sum()))
    if mode == "C":
        capture = independent_capture(img_fine4, cfg)
    else:
        scene = truth_img if mode == "A" else area_mean(img_fine4, FINE // 2)
        truth = TruthCameraModel(
            scale=cfg.scale * (1 if mode == "A" else 2), psf_sigma_lr=cfg.psf_sigma_lr_truth,
            read_noise_std=cfg.read_noise_std, shot_electrons=cfg.shot_electrons,
            saturation_level=cfg.saturation_level,
        )
        scene_target = Target(target.target_id, scene, mask, target.feature_count)
        capture = SimulatedRig(truth, scene_target, calibration_from_config(cfg)).capture_burst(
            lattice_schedule(cfg.frame_count, cfg.root_seed), root_seed=cfg.root_seed)
    model = EstimatedCameraModel(scale=cfg.scale, psf_sigma_lr=cfg.psf_sigma_lr_estimated)
    recon, _ = map_reconstruct(capture, model, regularization=cfg.regularization, max_iterations=cfg.max_iterations)
    thresholds = GateThresholds(cfg.gate_z2_threshold, min_valid_frames=max(4, cfg.frame_count // 4))
    gate = consistency_gate(recon, capture, model, thresholds)
    classes, conf = classify_provenance(capture, gate)
    return Scenario(cfg, target, capture, model, thresholds, recon, gate, classes, conf)


def eps_sweep(s: Scenario, proposal: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    err = np.abs(proposal - s.target.image)
    strong = classes >= 2
    out = {}
    for eps in EPSILONS:
        wrong = err > eps
        out[str(eps)] = {
            "wrong_fraction": round(float(wrong.mean()), 4),
            "unsupported_pass": round(float((strong & wrong).sum() / max(1, int(wrong.sum()))), 4),
            "precision_R": round(float((strong & ~wrong).sum() / max(1, int(strong.sum()))), 4),
        }
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
        res[name] = {"caught": round(float((classes[m] < 2).mean()), 3),
                     "rmse_vs_truth": round(float(np.sqrt(np.mean((p - s.target.image) ** 2))), 4)}
    return res


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_realesrgan(s: Scenario, exe: Path | None, workdir: Path) -> tuple[dict[str, Any], np.ndarray | None]:
    if exe is None:
        return {"status": "NOT RUN", "reason": "no --realesrgan executable supplied"}, None
    if not exe.exists():
        return {"status": "NOT RUN", "reason": f"executable not found: {exe}"}, None
    model = exe.parent / "models" / "realesrgan-x4plus.bin"
    frame = s.capture.frames[0]
    lr = np.clip(frame.data / np.float32(frame.params.exposure_scale), 0, 1)
    src, dst = (workdir / "lr.png").resolve(), (workdir / "sr.png").resolve()
    dst.unlink(missing_ok=True)
    Image.fromarray((lr * 255).astype(np.uint8), "L").convert("RGB").save(src)
    cmd = [str(exe), "-i", str(src), "-o", str(dst), "-n", "realesrgan-x4plus", "-s", "4"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=exe.parent)
    meta: dict[str, Any] = {
        "command": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-400:], "stderr_tail": proc.stderr[-400:],
        "exe_sha256": sha256_file(exe), "model_sha256": sha256_file(model) if model.exists() else "missing",
        "src_dims": list(lr.shape),
    }
    if proc.returncode != 0 or not dst.exists():
        return {"status": "ERROR", **meta}, None
    sr = np.asarray(Image.open(dst).convert("L"), dtype=np.float32) / 255.0
    meta["dst_dims"] = list(sr.shape)
    if sr.shape != s.target.image.shape:
        return {"status": "ERROR", "reason": "output shape mismatch", **meta}, None
    return {"status": "OK", **meta}, sr


def evaluate_proposals(s: Scenario, exe: Path | None, workdir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frame = s.capture.frames[0]
    lr = np.clip(frame.data / np.float32(frame.params.exposure_scale), 0, 1)
    bic = np.asarray(Image.fromarray((lr * 255).astype(np.uint8), "L").resize(
        s.target.image.shape[::-1], Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    tool, sr = run_realesrgan(s, exe, workdir)
    proposals: dict[str, np.ndarray] = {"map_baseline": s.reconstruction, "bicubic": bic}
    if sr is not None:
        proposals["realesrgan"] = sr
    out: dict[str, Any] = {"realesrgan_tool": tool}
    keep: dict[str, np.ndarray] = {}
    for label, prop in proposals.items():
        gate, classes, _ = s.evaluate(prop.astype(np.float32))
        out[label] = {**rates(s, prop, classes, None), "eps_sweep": eps_sweep(s, prop, classes),
                      "rmse_vs_truth": round(float(np.sqrt(np.mean((prop - s.target.image) ** 2))), 4)}
        keep[label] = prop
        keep[label + "_classes"] = classes
    return out, keep


def panel(s: Scenario, keep: dict[str, np.ndarray], path: Path, eps: float = 0.15) -> None:
    def g(a: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> Image.Image:
        return Image.fromarray((np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), "L").convert("RGB")

    prop = keep.get("realesrgan", keep["bicubic"])
    classes = keep.get("realesrgan_classes", keep["bicubic_classes"])
    err = np.abs(prop - s.target.image)
    wrong_r = (err > eps) & (classes >= 2)
    lr = s.capture.frames[0].data
    lr_up = np.kron(lr, np.ones((s.cfg.scale, s.cfg.scale), np.float32))
    ov = Image.alpha_composite(g(prop).convert("RGBA"), provenance_rgba(classes, alpha=110)).convert("RGB")
    mask = np.zeros((*wrong_r.shape, 3), np.uint8)
    mask[wrong_r] = (255, 40, 40)
    tiles = [g(s.target.image), g(lr_up), g(s.reconstruction), g(prop), g(err, 0, 0.3), ov, Image.fromarray(mask)]
    w, h = tiles[0].size
    out = Image.new("RGB", (len(tiles) * (w + 8), h), (16, 16, 16))
    for i, t in enumerate(tiles):
        out.paste(t, (i * (w + 8), 0))
    out.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--crop", action="append", default=[], help="stem=x0,y0,x1,y1")
    ap.add_argument("--realesrgan", type=Path)
    ap.add_argument("--output", type=Path, default=Path("artifacts/real_texture"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "synthetic_only": True,
        "statement": "User-supplied images are treated solely as synthetic scene textures; results refer "
                     "to simulated observations, not to the source files.",
        "panel_columns": ["truth", "LR frame 0 (nearest x4)", "MAP baseline", "proposal (Real-ESRGAN or bicubic)",
                          "|error| 0..0.3", "proposal + native provenance", "wrong(>0.15) AND R"],
    }
    if not args.images:
        report["status"] = "NOT RUN: no input images supplied"
    crops = {k: tuple(int(v) for v in c.split(",")) for k, c in (x.split("=", 1) for x in args.crop)}
    cfg = DemoConfig(max_iterations=24)
    fine_shape = (cfg.hr_shape[0] * FINE, cfg.hr_shape[1] * FINE)
    for path in args.images:
        if not path.exists():
            report[path.stem] = {"status": "NOT RUN", "reason": "input missing"}
            continue
        jobs: list[tuple[str, Any]] = [(path.stem, None)]
        if path.stem in crops:
            jobs.append((f"{path.stem}-crop", crops[path.stem]))
        for name, crop in jobs:
            img = load_gray(path, fine_shape, crop)
            for mode in ("A", "B", "C"):
                s = build(img, mode, cfg)
                row: dict[str, Any] = {"baseline": rates(s, s.reconstruction, s.provenance, None),
                                       "attacks": attacks(s)}
                for sig in (0, 2):
                    t = gaussian_blur_periodic(s.target.image, sig) if sig else s.target.image
                    row[f"map_within_0.05_sigma{sig}"] = round(float((np.abs(s.reconstruction - t) < 0.05).mean()), 4)
                if mode in ("B", "C"):
                    row["proposals"], keep = evaluate_proposals(s, args.realesrgan, args.output)
                    panel(s, keep, args.output / f"{name}_{mode}_panel.png")
                report[f"{name}/{mode}"] = row
                print(f"{name}/{mode} done", flush=True)
    (args.output / "real_texture.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output / 'real_texture.json'}")


if __name__ == "__main__":
    main()
