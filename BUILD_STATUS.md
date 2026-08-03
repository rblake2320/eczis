# UNFILED-IP
# ECZIS v0.1R1 Build Status

## Implemented and executed

- Independent truth and estimated camera models.
- Non-zero PSF, boundary, nonlinearity, illumination, and noise mismatch.
- Exact forward/adjoint inner-product test for the estimated linear operator.
- Five deterministic target generators.
- Simulated multi-frame capture rig and phase-diverse schedule.
- Bicubic, shift-and-add, and Tikhonov/Laplacian MAP reconstruction.
- Robust frame weighting and solver diagnostics.
- Heteroscedastic noise-normalized cycle residual gate.
- Detectable false-microglyph injection challenge.
- Conservative R/I/U classification; M is intentionally absent at HR because reconstructed HR pixels are not direct samples.
- Provenance-preserving tiled continuous renderer with scale demotion.
- Canonical artifact manifest, SHA-256 hashes, HMAC seal, ZIP evidence bundle, and chained JSONL ledger.
- Offline viewer with zoom, provenance overlay, residual overlay, and point inspector.
- Staged CLI plus one-command demonstration.
- 18 automated tests passing in the supplied execution environment.

## Implemented but not release-complete

- The five target families exist, but the current sealed demo uses the microglyph target only.
- Thresholding is fixed and adversarially tested, but a full calibration/development/locked-test matrix has not yet been executed.
- Same-platform deterministic behavior is implemented; Windows-vs-Linux numerical tolerance evidence still requires CI execution.

## Deferred by design

- Physical camera, XY stage, focus rail, and LED-ring drivers.
- Microscope ground-truth acquisition.
- Coordinate MLP/Fourier-feature field.
- Diffusion or foundation-model prior.
- DPAPI/CNG production key storage.
- Public-key signatures and trusted timestamps.
- Real-world, forensic, medical, government, legal, certification, or patentability claims.

## Validation performed here

- `pytest`: 18/18 passed.
- End-to-end synthetic demonstration: passed.
- Fresh bundle verification: passed.
- Wrong-key and artifact-mutation verification tests: passed.
- False-feature challenge demotion: passed on the registered demo case.

`mypy` and `ruff` are configured in `pyproject.toml`; they could not be installed in the isolated build container because outbound package resolution was unavailable. `Run-ECZIS.ps1` executes both after `uv sync` on a network-connected workstation.

## GitHub repository readiness

- Private-repository publication guard added.
- Cross-platform Windows/Ubuntu CI added.
- CODEOWNERS, contribution workflow, governance, issue templates, Dependabot, and roadmap added.
- Idempotent PowerShell publisher added for `rblake2320/eczis`.
- Local pytest, coverage, claims, IP, and compilation gates pass after repository hardening.
- GitHub-hosted `ruff` and `mypy` execution remains to be confirmed by the first connected CI run.
