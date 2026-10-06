# UNFILED-IP
"""ECZIS validation battery: sweeps that report three separate outcome rates.

  caught            fraction of planted false-detail pixels that are NOT labelled R/M
  false_demotion    fraction of accurately reconstructed pixels that are NOT labelled R/M
  unsupported_pass  fraction of wrong pixels (|proposal - truth| > 0.15) labelled R/M

Synthetic software reduction-to-practice only. Output goes to artifacts/battery/ (gitignored).
Usage: uv run python research/battery.py [--realesrgan PATH_TO_EXE]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eczis.bench.scenario import (
    TARGET_FAMILIES,
    Scenario,
    high_passed_glyph,
    null_space_detail,
    run_scenario,
    with_config,
)
from eczis.schemas import BoolArray, FloatArray, ProvenanceClass, UInt8Array

WRONG = 0.15
ACCURATE = 0.05


def _strong(classes: UInt8Array) -> BoolArray:
    return classes >= int(ProvenanceClass.R)


def rates(s: Scenario, proposal: FloatArray, classes: UInt8Array, planted: BoolArray | None) -> dict[str, Any]:
    err = np.abs(proposal - s.target.image)
    strong = _strong(classes)
    wrong, accurate = err > WRONG, err < ACCURATE
    out: dict[str, Any] = {
        "R_fraction": float(strong.mean()),
        "unsupported_pass": float((strong & wrong).sum() / max(1, int(wrong.sum()))),
        "false_demotion": float((~strong & accurate).sum() / max(1, int(accurate.sum()))),
        "genuine_feature_R": float(strong[s.target.feature_mask].mean()) if s.target.feature_mask.any() else None,
        "wrong_pixels": int(wrong.sum()),
    }
    if planted is not None:
        out["caught"] = float((~strong[planted]).mean())
    return out


def detectable_glyph(shape: tuple[int, int], amplitude: float) -> tuple[FloatArray, BoolArray]:
    h, w = shape
    mask = np.zeros(shape, dtype=np.bool_)
    y0, x0 = h // 2 - 10, w // 2 - 14
    mask[y0 : y0 + 22, x0 : x0 + 3] = True
    mask[y0 : y0 + 22, x0 + 18 : x0 + 21] = True
    mask[y0 + 9 : y0 + 13, x0 : x0 + 21] = True
    return (mask * np.float32(amplitude)).astype(np.float32), mask


def challenge_suite(s: Scenario) -> dict[str, Any]:
    shape = s.reconstruction.shape
    center = (shape[0] // 2, shape[1] // 2)
    res: dict[str, Any] = {"baseline": rates(s, s.reconstruction, s.provenance, None)}
    for name, (detail, mask) in {
        "detectable_glyph_0.72": detectable_glyph(shape, 0.72),
        "null_checker_0.35": null_space_detail(shape, center, amplitude=0.35, radius=8),
        "highpass_glyph_0.7": high_passed_glyph(shape, center, amplitude=0.7, model=s.model),
    }.items():
        proposal = np.clip(s.reconstruction + detail, -1.0, 2.0).astype(np.float32)
        _, classes, _ = s.evaluate(proposal)
        res[name] = rates(s, proposal, classes, mask)
    return res


def sweep(name: str, values: list[Any], key: str, family: str = "microglyph") -> list[dict[str, Any]]:
    rows = []
    for v in values:
        cfg = with_config(**{key: v})
        t0 = time.perf_counter()
        s = run_scenario(family, cfg)
        row = {"sweep": name, key: v, **challenge_suite(s), "seconds": round(time.perf_counter() - t0, 2)}
        rows.append(row)
        print(f"  {name} {key}={v} done ({row['seconds']}s)", flush=True)
    return rows


def realesrgan_challenge(exe: Path) -> dict[str, Any]:
    s = run_scenario("microglyph")
    frame = s.capture.frames[0]  # F000: dx=dy=0, aligned to the HR grid
    lr = np.clip(frame.data / np.float32(frame.params.exposure_scale), 0, 1)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "lr.png", Path(tmp) / "sr.png"
        Image.fromarray((lr * 255).astype(np.uint8), "L").convert("RGB").save(src)
        proc = subprocess.run(
            [str(exe), "-i", str(src), "-o", str(dst), "-n", "realesrgan-x4plus", "-s", "4"],
            capture_output=True, text=True, cwd=exe.parent,
        )
        if proc.returncode != 0 or not dst.exists():
            return {"status": "failed", "stderr": proc.stderr[-800:]}
        sr = np.asarray(Image.open(dst).convert("L"), dtype=np.float32) / 255.0
    bic = np.asarray(
        Image.fromarray((lr * 255).astype(np.uint8), "L").resize(sr.shape[::-1], Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0
    out: dict[str, Any] = {"status": "ok", "map_reference": rates(s, s.reconstruction, s.provenance, None)}
    for label, proposal in {"realesrgan_x4plus": sr, "bicubic_control": bic}.items():
        gate, classes, _ = s.evaluate(proposal.astype(np.float32))
        out[label] = {
            **rates(s, proposal, classes, None),
            "gate_pass_fraction": float(gate.pass_mask_hr.mean()),
            "rmse_vs_truth": float(np.sqrt(np.mean((proposal - s.target.image) ** 2))),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--realesrgan", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/battery"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"synthetic_only": True, "thresholds": {"wrong": WRONG, "accurate": ACCURATE}}

    print("families", flush=True)
    report["families"] = {f: challenge_suite(run_scenario(f)) for f in TARGET_FAMILIES}
    print("sweeps", flush=True)
    report["mismatch_psf"] = sweep("mismatch", [0.5, 0.6, 0.7, 0.82, 0.95, 1.1], "psf_sigma_lr_estimated")
    report["read_noise"] = sweep("snr", [0.002, 0.004, 0.008, 0.016, 0.032], "read_noise_std")
    report["gate_threshold"] = sweep("threshold", [2.0, 4.0, 8.0, 16.0, 32.0], "gate_z2_threshold")
    report["frame_count"] = sweep("frames", [4, 9, 16, 25], "frame_count")

    print("amplitude", flush=True)
    s = run_scenario("microglyph")
    amp = []
    for a in [0.02, 0.05, 0.1, 0.2, 0.4, 0.72]:
        detail, mask = detectable_glyph(s.reconstruction.shape, a)
        _, classes, _ = s.evaluate(s.reconstruction + detail)
        amp.append({"amplitude": a, **rates(s, s.reconstruction + detail, classes, mask)})
    report["detectable_amplitude"] = amp

    if args.realesrgan:
        print("realesrgan", flush=True)
        report["realesrgan"] = realesrgan_challenge(args.realesrgan)
    else:
        report["realesrgan"] = {"status": "not_run"}
    report["seedvr2"] = {"status": "not_run", "reason": "requires ComfyUI server + weights; not attempted"}

    (args.output / "battery.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output / 'battery.json'}")


if __name__ == "__main__":
    main()
