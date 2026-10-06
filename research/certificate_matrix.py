# UNFILED-IP
"""Feature x acquisition matrix for the EXPERIMENTAL support field and certificates.

Synthetic software reduction-to-practice only. Rows: stroke features; columns: acquisitions.
"digital_macro_correct" enlarges frame 0 (a deterministic function of data already held) and
uses the correct correlated noise model, so it adds no information. "digital_macro_naive" treats
the enlarged pixels as new independent measurements, the error that fakes certainty.
Output: artifacts/certificates/matrix.json (gitignored).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from eczis.camera.estimated_model import EstimatedCameraModel
from eczis.experimental.certificates import Question, certify
from eczis.experimental.support import (
    Measurement,
    SupportField,
    build_operator,
    noise_variance,
    support_field,
)
from eczis.rig.schedule import lattice_schedule
from eczis.schemas import ObservationParams, ProvenanceClass

N, SHOT, READ = 32, 700.0, 0.004


def scene(kind: str, present: bool) -> tuple[np.ndarray, Question]:
    x = np.full((N, N), 0.2)
    for r in (8, 16, 24):
        x[r : r + 2, 10:22] += 0.5
    x[8:26, 20:22] += 0.5
    width = {"stroke_1px": 1, "stroke_2px": 2, "stroke_broad": 6}[kind]
    if present:
        x[8:26, 10 : 10 + width] += 0.5 if kind != "stroke_broad" else 0.25
    m = np.zeros((N, N))
    m[11:23, 10 : 10 + width] = 1.0
    contrast = 0.5 if kind != "stroke_broad" else 0.25
    return np.clip(x, 0, 1), Question(f"{kind}_present", (m / m.sum()).ravel(), 0.2 + contrast / 2 + 0.02)


def cap(scale: int, psf: float, phases: list[tuple[float, float]], tag: str) -> list[Measurement]:
    model = EstimatedCameraModel(scale=scale, psf_sigma_lr=psf)
    return [Measurement(model, ObservationParams(f"{tag}{i}", dx, dy, 0.0, 1.0, 0.0, 0.0, 700 + i))
            for i, (dx, dy) in enumerate(phases)]


def enlarge_matrix(lr: int, factor: int) -> np.ndarray:
    """Bilinear digital enlargement of an lr x lr frame (deterministic, rank <= lr*lr)."""
    def axis(n_in: int) -> np.ndarray:
        n_out = n_in * factor
        m = np.zeros((n_out, n_in))
        for o in range(n_out):
            pos = min(max((o + 0.5) / factor - 0.5, 0.0), n_in - 1.0)
            i0 = int(np.floor(pos))
            f = pos - i0
            m[o, i0] += 1 - f
            if f > 0:
                m[o, min(i0 + 1, n_in - 1)] += f
        return m

    a = axis(lr)
    return np.kron(a, a)


def band_summary(sf: SupportField) -> dict[str, float]:
    return {f"L{lv}": round(float(np.median(sf.amplitude[sf.level == lv])), 4) for lv in (1, 2, 3, 4)}


def main() -> None:
    out_dir = Path("artifacts/certificates")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ops = [Measurement(EstimatedCameraModel(scale=4, psf_sigma_lr=0.70), o)
                for o in lattice_schedule(16, 20260803).observations]
    a_base = build_operator(base_ops, (N, N))
    extras = {
        "redundant_x2": a_base.copy(),
        "closer_x2": build_operator(cap(2, 0.5, [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)], "C"), (N, N)),
        "physical_macro_x1": build_operator(cap(1, 0.6, [(0, 0), (0.5, 0.5)], "M"), (N, N)),
    }
    t_zoom = enlarge_matrix(8, 4)  # frame-0 digital enlargement to the HR grid
    report: dict[str, Any] = {"synthetic_only": True, "matrix": {}}
    rng_seed = 100
    for kind in ("stroke_1px", "stroke_2px", "stroke_broad"):
        for present in (True, False):
            truth, q = scene(kind, present)
            true_answer = float(q.weights @ truth.ravel())
            row: dict[str, Any] = {"true_answer": round(true_answer, 4), "threshold": q.threshold}
            rng = np.random.default_rng(rng_seed)
            rng_seed += 1
            y0 = a_base @ truth.ravel()
            v0 = noise_variance(y0, SHOT, READ)
            y0 = y0 + rng.normal(0, np.sqrt(v0))
            sf0 = support_field(a_base, y0, v0, (N, N))
            base_sd = float(np.sqrt(q.weights @ sf0.cov @ q.weights))
            cols: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {"base": (a_base, y0, v0)}
            for name, rows in extras.items():
                y1 = rows @ truth.ravel()
                v1 = noise_variance(y1, SHOT, READ)
                y1 = y1 + rng.normal(0, np.sqrt(v1))
                cols[name] = (np.vstack([a_base, rows]), np.concatenate([y0, y1]), np.concatenate([v0, v1]))
            # Digital macro: z = T y0[frame 0]. Correct model: its covariance T S T^T is singular and
            # fully correlated with y0, so the joint data carry exactly the information of y0.
            f0 = slice(0, 64)
            z = t_zoom @ y0[f0]
            a_z = t_zoom @ a_base[f0]
            cols["digital_macro_correct"] = (a_base, y0, v0)  # information-equivalent by construction
            cols["digital_macro_naive"] = (np.vstack([a_base, a_z]), np.concatenate([y0, z]),
                                           np.concatenate([v0, noise_variance(z, SHOT, READ)]))
            for name, (a, y, v) in cols.items():
                sf = support_field(a, y, v, (N, N))
                c = certify(sf, q)
                sd = float(np.sqrt(q.weights @ sf.cov @ q.weights))
                row[name] = {
                    "decision": c.decision.value, "answer": c.answer,
                    "interval": [round(c.lower, 4), round(c.upper, 4)],
                    "contains_truth": bool(c.lower <= true_answer <= c.upper),
                    "answer_sd": round(sd, 4), "sd_reduction_vs_base": round(1 - sd / base_sd, 4),
                    "band_amplitude": band_summary(sf),
                    "R_bands": sorted({int(lv) for lv in np.unique(sf.level)
                                       if np.all(sf.coefficient_class[sf.level == lv] == int(ProvenanceClass.R))}),
                }
            report["matrix"][f"{kind}/{'present' if present else 'absent'}"] = row
            print(f"{kind}/{'present' if present else 'absent'} done", flush=True)
    (out_dir / "matrix.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'matrix.json'}")


if __name__ == "__main__":
    main()
