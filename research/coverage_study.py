# UNFILED-IP
"""Coverage study for EXPERIMENTAL answer-certificate intervals (issue #10).

Per cell (feature x truth class x acquisition x prior strength) this draws N independent noise
realisations and reports coverage of kappa=3 intervals, bias, RMSE, mean reported s.e., empirical
s.d. of the estimate, z statistics, width, false-certification and abstention rates.

Interval variants:
  B  posterior mean +/- 3 posterior s.d. (the current certificate)
  C  B widened by a worst-case regularisation-bias bound: |bias| <= ||C g||_1 / s_p^2 * max|x - m|
     with x in [0, 1] (box prior knowledge, no truth used)
Prior strengths s_p = 1 (current), 10, 1000 (near-unregularised, variant "A").
The posterior covariance does not depend on y here, so it is computed once per cell.
Synthetic only; output artifacts/certificates/coverage.json (gitignored).
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from certificate_matrix import READ, SHOT, N, cap, scene  # noqa: E402

from eczis.camera.estimated_model import EstimatedCameraModel  # noqa: E402
from eczis.experimental.support import Measurement, build_operator, noise_variance  # noqa: E402
from eczis.rig.schedule import lattice_schedule  # noqa: E402

KAPPA = 3.0


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def main(trials: int = 2000) -> None:
    out_dir = Path("artifacts/certificates")
    out_dir.mkdir(parents=True, exist_ok=True)
    a_base = build_operator([Measurement(EstimatedCameraModel(scale=4, psf_sigma_lr=0.70), o)
                             for o in lattice_schedule(16, 20260803).observations], (N, N))
    acq = {
        "base": a_base,
        "closer": np.vstack([a_base, build_operator(cap(2, 0.5, [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)], "C"), (N, N))]),
        "macro": np.vstack([a_base, build_operator(cap(1, 0.6, [(0, 0), (0.5, 0.5)], "M"), (N, N))]),
    }
    report: dict[str, Any] = {"synthetic_only": True, "trials_per_cell": trials, "kappa": KAPPA,
                              "nominal_coverage": 0.9973, "cells": {}}
    for kind in ("stroke_broad", "stroke_2px"):
        for present in (True, False):
            truth, q = scene(kind, present)
            g = q.weights
            theta = float(g @ truth.ravel())
            for acq_name, a in acq.items():
                y_clean = a @ truth.ravel()
                var = noise_variance(y_clean, SHOT, READ)
                d = 1.0 / var
                ata = (a.T * d) @ a
                rng = np.random.default_rng(zlib.crc32(f"{kind}|{present}|{acq_name}".encode()))
                noise = rng.standard_normal((trials, y_clean.size)) * np.sqrt(var)
                ys = y_clean[None, :] + noise
                for sp in (1.0, 10.0, 1000.0):
                    cov = np.linalg.inv(ata + np.eye(a.shape[1]) / sp**2)
                    cg = cov @ g
                    se = float(np.sqrt(g @ cg))
                    m = ys.mean(axis=1)  # prior mean = mean(y), as in support_field
                    est = (ys * d) @ (a @ cg) + m * cg.sum() / sp**2
                    bias_bound = float(np.abs(cg).sum() / sp**2 * np.max(np.maximum(m, 1 - m)))
                    row: dict[str, Any] = {"theta": round(theta, 4), "threshold": q.threshold,
                                           "reported_se": round(se, 5), "bias_bound_C": round(bias_bound, 5)}
                    err = est - theta
                    row.update({"bias": round(float(err.mean()), 5), "rmse": round(float(np.sqrt((err**2).mean())), 5),
                                "empirical_sd": round(float(est.std()), 5),
                                "z_mean": round(float((err / se).mean()), 3), "z_sd": round(float((err / se).std()), 3),
                                "max_abs_z": round(float(np.abs(err / se).max()), 2)})
                    for variant, half in (("B", KAPPA * se), ("C", KAPPA * se + bias_bound)):
                        lo, hi = est - half, est + half
                        covered = (lo <= theta) & (theta <= hi)
                        certified_above, certified_below = lo > q.threshold, hi < q.threshold
                        wrong = certified_below if present else certified_above
                        k = int(covered.sum())
                        row[variant] = {
                            "coverage": round(k / trials, 4), "coverage_ci95": [round(v, 4) for v in clopper_pearson(k, trials)],
                            "misses": trials - k, "width": round(2 * half, 5),
                            "false_certification_rate": round(float(wrong.mean()), 4),
                            "abstention_rate": round(float((~certified_above & ~certified_below).mean()), 4),
                        }
                    report["cells"][f"{kind}/{'present' if present else 'absent'}/{acq_name}/sp{sp:g}"] = row
            print(f"{kind}/{'present' if present else 'absent'} done", flush=True)
    (out_dir / "coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'coverage.json'}")


if __name__ == "__main__":
    main()
