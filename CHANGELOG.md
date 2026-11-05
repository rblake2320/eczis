# UNFILED-IP
# Changelog

All notable ECZIS changes are recorded here. The project follows semantic versioning once a
stable public API exists; pre-1.0 releases may contain intentional interface changes.

## [0.1.1] - 2026-11-05

### Fixed (P0 — evidence honesty)

- **Non-identifiable detail was labeled R** (stop condition 6). 99.87 % of demo pixels were R at
  11.9 dB PSNR. R now requires supported-band identifiability; see ADR-0002.
- **K-1 attack passed the gate**: one contradicting frame was diluted by the mean. Added a
  per-frame contradiction criterion (threshold 24, registered from clean runs).
- `Run-ECZIS.ps1 -RunStaticAnalysis` ran `mypy --strict src tests`, which failed; tests are now strict-clean.
- `BUILD_STATUS.md` claimed ruff/mypy could not run and listed 18 tests; corrected.
- `AGENTS.md` referenced a gitignored `evidence/` path that was not ignored; now ignored and enforced.

### Added

- Ledger: cross-process lock, torn-tail detection/repair, HMAC-sealed head anchor.
- Keys: exclusive create, DPAPI wrapping on Windows, `eczis key create`.
- Bundle: replay-environment metadata, deterministic archive bytes, schema-major rejection,
  unsafe/duplicate member rejection, canonical-manifest enforcement, atomic manifest writes.
- `camera/noise.py`, `camera/cfa.py`, `versioning.py`, `atomic.py`.
- Single-frame deconvolution baseline; five-family benchmark matrix with PSF-mismatch arm,
  bootstrap CIs, false-R (band-limited and full-band), ECE, calibration-split calibrator,
  split-leakage guard. `eczis bench`, `eczis export`, `--config`, `target create --family`.
- Optional Fourier-feature `CoordinateFieldRenderer` (demote-only).
- `tools/claims_check.py --junit` (executed mode), `tools/verify_release.py`,
  `tools/evidence_guard.py`, `tools/hooks/` + `tools/install_hooks.py`, `tools/regen_golden.py`.
- Golden fixtures, adversarial suites for gate/seal/ledger, CLI/tool tests. 18 -> 140 tests.
- `RELEASE_CLAIMS.yaml`: 4 -> 15 claims. ADR-0002. Internal `docs/PATENT_CLAIM_SEEDS.md`.

### Fixed (found by the first Windows CI run)

- Manifest order was platform-dependent: `Path` sorting is case-insensitive on Windows, so
  mixed-case artifact names produced a non-canonical manifest that the verifier rejected.
  Manifests now sort by POSIX relative string (code-point order) on every OS.
- The backslash-path attack test was defanged on Windows (`ZipInfo` rewrites `\\` to `/`);
  the raw name is now set explicitly.

### Changed

- `classify_provenance` now requires `proposal_hr` and `model` (keyword-only); CLI
  `provenance` requires `--reconstruction`.
- Manifest schema 1.0 -> 1.1 (adds `environment`); verifier accepts any 1.x.
- CI: evidence guard, strict mypy on tests, JUnit + release verification, coverage floor 80 %,
  manual locked-benchmark job.

## [0.1.0] - 2026-08-03

### Added

- Independent truth and estimated camera models.
- Synthetic multi-frame capture and procedural target library.
- Bicubic, shift-and-add, and regularized MAP reconstruction.
- Noise-normalized cycle-consistency gate.
- Conservative M/R/I/U provenance accounting.
- Provenance-preserving renderer and offline forensic viewer.
- Canonical manifest, SHA-256 hashes, HMAC seal, and chained ledger.
- Staged CLI, PowerShell runner, adversarial challenge, and automated tests.
- GitHub CI, private-repository publication workflow, issue templates, and development roadmap.
