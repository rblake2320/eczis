# UNFILED-IP
"""Clean up a burst of frames the evidence-aware way: register, fuse, and label what is supported.

    uv run python tools/burst_enhance.py artifacts/earscan/<time>            # uses frames.npz
    uv run python tools/burst_enhance.py <folder> --ref 60 --tol 0.06 --deblur 1.2

Every frame is aligned to the reference frame (ECC affine, coarse-to-fine), the aligned stack is
averaged (noise falls as 1/sqrt(N)), and each pixel is labelled by how many aligned frames agree
with the fused value within --tol:

    R  >= 2/3 of the frames agree       the fused pixel is multiply supported
    I  >= 1/3 agree                     partially supported (motion blur, parallax, edges)
    U  fewer                            unsupported: shown from the reference frame only

Outputs in <folder>/enhance/: fused.png (R/I pixels fused, U from the reference), support.png
(blue R / amber I / grey U), fused_deblur.png (optional unsharp/Richardson-Lucy, which is a
RE-RENDERING of the fused image and is labelled as such: it adds no evidence), report.json.
No generative model is used; nothing is drawn that the frames did not contain. EXPERIMENTAL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

R_FRAC, I_FRAC = 2 / 3, 1 / 3


def align(ref: np.ndarray, img: np.ndarray, iters: int = 60) -> tuple[np.ndarray, float, bool]:
    """ECC affine alignment of img onto ref (both float32 0..1); returns (warped, correlation, ok)."""
    import cv2

    warp = np.eye(2, 3, dtype=np.float32)
    small_ref = cv2.resize(ref, None, fx=0.5, fy=0.5)
    small_img = cv2.resize(img, None, fx=0.5, fy=0.5)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5)
    try:
        cc, warp = cv2.findTransformECC(small_ref, small_img, warp, cv2.MOTION_AFFINE, crit, None, 5)  # type: ignore[call-overload]
        warp[:, 2] *= 2.0
        cc, warp = cv2.findTransformECC(ref, img, warp, cv2.MOTION_AFFINE, crit, None, 5)  # type: ignore[call-overload]
    except cv2.error:
        return img, 0.0, False
    h, w = ref.shape
    out = cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    return np.asarray(out, dtype=np.float32), float(cc), True


SATURATED = 0.98  # clipped-bright pixels carry no scene information: never evidence, whatever the agreement


def fuse(frames: list[np.ndarray], ref_idx: int, tol: float, min_cc: float = 0.9) -> dict[str, np.ndarray | int | float]:
    ref = np.asarray(frames[ref_idx], np.float32)
    stack, ccs, rejected = [ref], [], 0
    for i, f in enumerate(frames):
        if i == ref_idx:
            continue
        w, cc, ok = align(ref, np.asarray(f, np.float32))
        if ok and cc >= min_cc:
            stack.append(w)
            ccs.append(cc)
        else:
            rejected += 1
    arr = np.stack(stack)
    mean = arr.mean(axis=0)
    agree = (np.abs(arr - mean[None]) <= tol).sum(axis=0)
    n = arr.shape[0]
    support = np.full(mean.shape, "U", dtype="<U1")
    support[agree >= I_FRAC * n] = "I"
    support[agree >= R_FRAC * n] = "R"
    saturated = (arr >= SATURATED).mean(axis=0) > 0.5  # clipped in most aligned frames
    support[saturated] = "U"
    fused = np.where(support == "U", ref, mean)
    return {"fused": fused, "support": support, "agree": agree, "n_aligned": n, "n_total": len(frames), "n_rejected": rejected,
            "cc_median": float(np.median(ccs)) if ccs else 0.0, "saturated_frac": float(saturated.mean()), "ref": ref}


def deblur(img: np.ndarray, sigma: float, iters: int = 12) -> np.ndarray:
    """Richardson-Lucy with a Gaussian PSF of the given sigma: a re-rendering, not new evidence."""
    from scipy.ndimage import gaussian_filter

    est = img.copy()
    for _ in range(iters):
        conv = gaussian_filter(est, sigma)
        ratio = img / np.maximum(conv, 1e-4)
        est = est * gaussian_filter(ratio, sigma)
    return np.asarray(np.clip(est, 0, 1), dtype=np.float32)


def save(path: Path, img: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--ref", type=int, default=None, help="reference frame index (default: middle)")
    ap.add_argument("--tol", type=float, default=0.06, help="agreement tolerance in intensity units")
    ap.add_argument("--deblur", type=float, default=0.0, help="Gaussian PSF sigma for Richardson-Lucy (0 = off)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min-cc", type=float, default=0.9, help="reject frames whose alignment correlation is below this")
    a = ap.parse_args()
    frames = [np.asarray(f, np.float32) for f in np.load(a.folder / "frames.npz")["frames"][:: a.stride]]
    ref_idx = a.ref if a.ref is not None else len(frames) // 2
    res = fuse(frames, ref_idx, a.tol, a.min_cc)
    out = a.folder / "enhance"
    out.mkdir(exist_ok=True)
    fused, support = res["fused"], res["support"]
    assert isinstance(fused, np.ndarray) and isinstance(support, np.ndarray)
    save(out / "reference.png", res["ref"])  # type: ignore[arg-type]
    save(out / "fused.png", fused)
    rgb = np.zeros(fused.shape + (3,), np.float32)
    rgb[support == "R"] = (37 / 255, 99 / 255, 235 / 255)
    rgb[support == "I"] = (245 / 255, 158 / 255, 11 / 255)
    rgb[support == "U"] = (120 / 255, 120 / 255, 120 / 255)
    from PIL import Image

    Image.fromarray((rgb * 255).astype(np.uint8)).save(out / "support.png")
    overlay = 0.6 * np.repeat(fused[..., None], 3, axis=2) + 0.4 * rgb
    Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(out / "fused_support_overlay.png")
    report: dict[str, object] = {
        "frames": res["n_total"], "aligned": res["n_aligned"], "rejected_low_alignment": res["n_rejected"], "min_cc": a.min_cc,
        "saturated_fraction_forced_U": round(float(res["saturated_frac"]), 4), "reference_index": ref_idx, "tol": a.tol,
        "alignment_cc_median": round(float(res["cc_median"]), 3),
        "support_fractions": {c: round(float((support == c).mean()), 4) for c in ("R", "I", "U")},
        "noise_sd_reference": round(float(np.std(res["ref"] - fused)), 4),
        "files": {"fused": str(out / "fused.png"), "support": str(out / "support.png"), "reference": str(out / "reference.png")},
        "notes": ["fused = mean of aligned frames where R/I; the reference frame where U",
                  "support counts aligned frames agreeing with the fused value within tol: evidence, not confidence",
                  "clipped-bright (specular) pixels are U regardless of agreement: a hotspot agreeing with itself is not evidence",
                  "no generative model; nothing drawn that the frames did not contain"],
    }
    if a.deblur > 0:
        db = deblur(fused, a.deblur)
        save(out / "fused_deblur.png", db)
        report["files"]["deblur"] = str(out / "fused_deblur.png")  # type: ignore[index]
        report["notes"].append(f"fused_deblur.png: Richardson-Lucy, Gaussian sigma {a.deblur}: a RE-RENDERING of fused.png, adds no evidence")  # type: ignore[attr-defined]
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
