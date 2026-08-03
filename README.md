# UNFILED-IP
# ECZIS v0.1R1 — Executable Software Proof

**Owner:** Ron Blake  
**Disclosure status:** UNFILED — keep the repository private.  
**Claim boundary:** **Infinitely sampleable output, finite and explicitly classified evidence.**

This repository contains the first runnable ECZIS vertical slice:

1. deterministic procedural microglyph ground truth;
2. an independent `TruthCameraModel` that creates noisy RAW-like burst observations;
3. a deliberately imperfect `EstimatedCameraModel` used for reconstruction and gating;
4. bicubic, registered shift-and-add, and regularized MAP reconstruction;
5. a noise-normalized cycle-consistency gate;
6. conservative M/R/I/U provenance;
7. a provenance-preserving continuous renderer;
8. hash-chained ledger and HMAC-sealed evidence bundle;
9. mutation verification;
10. an offline forensic viewer; and
11. an automated adversarial test that injects a false microglyph and verifies demotion.

This is a **synthetic software reduction-to-practice**, not evidence of physical-camera performance.

## Windows: one-command build and demonstration

From PowerShell 7 in the extracted repository:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\Run-ECZIS.ps1 -OpenViewer
```

The script installs the environment through `uv`, runs the automated tests, creates the synthetic evidence bundle, verifies it, and opens the offline viewer. Add `-RunStaticAnalysis` after `mypy` and `ruff` have resolved on the workstation.

## Direct commands

```powershell
uv sync --python 3.12
uv run eczis doctor
uv run pytest
uv run eczis demo --output .\artifacts\demo --seed 20260803 --frames 16 --iterations 36
uv run eczis verify --bundle .\artifacts\demo\eczis_demo.eczis --key .\artifacts\demo\DEMO_ONLY.key
```

## Staged workflow

```powershell
uv run eczis target create --output .\artifacts\staged\target --seed 20260803
uv run eczis capture simulate --target .\artifacts\staged\target --output .\artifacts\staged\capture.npz --seed 20260803 --frames 16
uv run eczis reconstruct --capture .\artifacts\staged\capture.npz --output .\artifacts\staged\recon --iterations 36
uv run eczis gate --capture .\artifacts\staged\capture.npz --reconstruction .\artifacts\staged\recon\map.npy --output .\artifacts\staged\gate --threshold 8
uv run eczis provenance --capture .\artifacts\staged\capture.npz --gate .\artifacts\staged\gate --output .\artifacts\staged\provenance
uv run eczis render --reconstruction .\artifacts\staged\recon\map.npy --gate .\artifacts\staged\gate --provenance .\artifacts\staged\provenance --output .\artifacts\staged\viewer.html
```

## Expected demo artifacts

```text
artifacts/demo/
├── ECZIS_VIEWER.html
├── REPORT.md
├── RUN_SUMMARY.json
├── verification.json
├── DEMO_ONLY.key
├── KEY_WARNING.txt
├── eczis_demo.eczis
└── artifacts/
    ├── truth.png
    ├── raw_burst.png
    ├── bicubic.png
    ├── shift_add.png
    ├── map_reconstruction.png
    ├── cycle_residual.png
    ├── provenance.png
    ├── challenged_candidate.png
    ├── challenge_residual.png
    ├── challenge_provenance.png
    ├── benchmark.json
    ├── ledger.jsonl
    ├── manifest.json
    └── seal.hmac
```

## Honest limitations

The current build does not claim a physical rig, real optical gains, a learned coordinate MLP, universal false-detail detection, DPAPI-backed production key management, public-key nonrepudiation, forensic admissibility, or patentability. See `BUILD_STATUS.md` for exact completion status.

## Private GitHub development

This source must remain in a private repository while `IP_STATUS.yaml` declares
`public_disclosure_allowed: false`.

To create and configure `rblake2320/eczis` from Windows in one guarded operation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\Publish-ECZIS-To-GitHub.ps1
```

The publisher refuses public repositories, blocks generated keys and evidence bundles, installs
missing `git`, `gh`, and `uv` tools through `winget`, creates `uv.lock`, executes the release gates,
creates the private repository, pushes `main`, configures topics and labels, creates the initial
backlog, and attempts to enable branch protection.

Development changes must use issues and pull requests after bootstrap. See `CONTRIBUTING.md`,
`GOVERNANCE.md`, and `docs/ROADMAP.md`.

## License posture

ECZIS is proprietary and all rights are reserved. Repository access does not grant permission to
publish, distribute, sublicense, or reuse the implementation. See `LICENSE`.
