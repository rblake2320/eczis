# UNFILED-IP
# ECZIS v0.1.1 Build Status

> Infinitely sampleable output, finite and explicitly classified evidence.
> Evidence level: synthetic software reduction-to-practice. No physical-camera claim.

## Verified in this build (Linux container; Windows leg runs in CI)

| Gate | Result |
|---|---|
| `uv run pytest` | 139 passed, 1 skipped (`platform-not-applicable`: DPAPI round-trip, Windows-only) |
| Coverage (`--cov=eczis`) | 95 % (CI floor raised 70 -> 80 %) |
| `ruff check .` | clean |
| `mypy --strict src tools tests` | clean (78 files) |
| `tools/ip_gate.py` | PASS |
| `tools/evidence_guard.py` | PASS |
| `tools/claims_check.py --junit` | PASS, executed mode: 15 claims, 47 mapped tests all executed and passed |
| `tools/verify_release.py` | PASS: release manifest with frozen test count + demo bundle verification |
| End-to-end demo + fresh bundle verification | PASS |

## Implemented

- Independent truth and estimated camera models (behavioural difference tested, not just class identity).
- Shared Poisson-Gaussian noise model (`camera/noise.py`) used by simulator and gate; explicit CFA geometry (`camera/cfa.py`, monochrome demo).
- Exact forward/adjoint identity for the estimated operator.
- Five deterministic target families, all run through the pipeline and the benchmark matrix.
- Baselines: bicubic, **single-frame deconvolution**, shift-and-add, Tikhonov/Laplacian MAP.
- Gate: region mean z^2 **plus per-frame contradiction criterion** (K-1 attack closed; ADR-0002).
- Provenance: **supported-band identifiability** required for R (stop condition 6 enforced; ADR-0002). M never emitted for HR.
- Renderers: deterministic tiled raster (baseline) and optional Fourier-feature coordinate field; both demote-only (property-tested).
- Seal: canonical manifest with replay-environment metadata, deterministic ZIP, HMAC, schema-major rejection, path-traversal/duplicate-member rejection, atomic sealing.
- Ledger: cross-process lock, torn-tail detection + quarantine repair, HMAC head anchor (detects truncation / history rewrite).
- Keys: exclusive-create (never overwrite), 0600 on POSIX, DPAPI wrapping on Windows, minimum length.
- Benchmark harness: split leakage guard, family x seed matrix, PSF-mismatch arm, bootstrap CIs, ECE + calibration-split calibrator.
- CLI: `doctor demo bench export key target capture reconstruct gate provenance render seal verify`, `--config` JSON (unknown keys fail closed), structured FAIL output.
- Tooling: executed claims gate, release verifier, evidence/personal-media guard, pre-commit / pre-push hooks.
- Golden fixtures: exact serialization digests, tolerance-bounded numerics.

## Honest limits (measured, reported, not claimed)

- R asserts the supported-band component only; full-band false-R is reported alongside.
- Gate detection rate for injected detail is SNR-dependent (coin at 250 e-: 0.32); injected R/M stays 0 in every tested regime.
- Shift-and-add does not beat bicubic on every family (see `docs/BENCHMARK_RESULTS.md`).
- Windows-vs-Linux numerics are asserted by golden tolerances; first Windows CI run on this branch is the evidence.

## Deferred by design

- Physical camera, XY stage, focus rail, LED-ring drivers; microscope ground truth.
- Diffusion or foundation-model priors.
- Public-key signatures and trusted timestamps (HMAC proves integrity only to a key holder).
- Real-world, forensic, medical, government, legal, certification, or patentability claims.
