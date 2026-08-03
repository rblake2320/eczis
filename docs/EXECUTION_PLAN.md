# UNFILED-IP
# ECZIS v0.1R1 — Evidence-Aware Continuous Zoom Imaging System
## Execution Plan and Acceptance Contract

**Owner:** Ron Blake  
**Repository:** `rblake2320/eczis` — private until filing authorization  
**Primary platform:** Windows 11, PowerShell 7, Python 3.12, `uv`  
**Quality bar:** deterministic, typed, test-gated, evidence-bounded, reproducible  
**Public-claim status:** prohibited until patent-disclosure gate is cleared

---

## 1. Build Decision

### Buildable now

ECZIS v0.1R1 is buildable as a complete software proof of concept containing:

- an independent synthetic truth generator;
- a deliberately imperfect reconstruction camera model;
- simulated capture schedules and RAW-like observations;
- multi-frame registration and super-resolution reconstruction;
- noise-aware cycle-consistency scoring;
- M/R/I/U provenance classification;
- a continuous zoom/rendering API;
- deterministic evidence manifests, integrity sealing, and a hash-chained ledger;
- benchmark suites that measure recovery, hallucination, calibration, and robustness;
- a local forensic viewer; and
- versioned interfaces for a later physical capture rig.

### Not buildable or provable in software alone

The following must not be claimed as completed by v0.1R1:

- construction or validation of the physical camera/stage/illumination rig;
- real-world optical resolution gains;
- universal hallucination detection;
- infinite factual resolution;
- forensic admissibility, legal nonrepudiation, or patentability;
- large diffusion-prior training; or
- performance on uncontrolled handheld photographs.

### Mandatory scientific correction

The truth simulator and reconstruction model must be separate implementations. Using one identical camera model to both synthesize observations and validate reconstruction creates an **inverse crime** and can make a flawed reconstruction appear correct.

ECZIS therefore uses:

1. `TruthCameraModel` — richer, hidden from reconstruction, and used only to create benchmark observations.
2. `EstimatedCameraModel` — calibrated from observations and used by reconstruction and the consistency gate.
3. A registered model-mismatch matrix — PSF error, chromatic shift, sensor nonlinearity, noise mismatch, saturation, and registration error.

No benchmark result is valid unless it passes under non-zero model mismatch.

---

## 2. Non-Negotiable Claims Language

Use only:

> **Infinitely sampleable output, finite and explicitly classified evidence.**

Never use these unqualified phrases:

- infinite resolution;
- infinite detail;
- perfect reconstruction;
- hallucination-free;
- forensic-grade;
- ground-truth recovery from one ordinary image.

Every release claim must map to one or more passing acceptance tests and a sealed test-result digest.

---

## 3. Scope Freeze

### Required in v0.1R1

1. Versioned data contracts and coordinate conventions.
2. Independent truth and estimated camera models.
3. Five deterministic procedural targets.
4. Simulated capture rig and serializable capture schedules.
5. Registration, shift-and-add, and regularized MAP reconstruction.
6. Noise-normalized cycle-consistency gate.
7. Conservative M/R/I/U provenance engine.
8. Continuous renderer with provenance-preserving queries.
9. Canonical evidence manifest, per-artifact hashes, HMAC seal, and ledger.
10. Benchmark harness with calibration/test separation and adversarial mismatch cases.
11. Offline forensic viewer.
12. Physical-rig integration contract and calibration protocol.

### Explicitly deferred

- diffusion or foundation-model priors;
- TinyCNN training as a release dependency;
- Fourier ptychography;
- event-camera support;
- quantum or metasurface imaging;
- real camera drivers;
- cloud services;
- multi-user authorization;
- medical, legal, government, or forensic certification.

---

## 4. Repository Layout

```text
eczis/
├── .github/workflows/ci.yml
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
├── IP_STATUS.yaml
├── RELEASE_CLAIMS.yaml
├── SECURITY.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CONVENTIONS.md
│   ├── DATA_CONTRACTS.md
│   ├── PROVENANCE_CLASSES.md
│   ├── VALIDATION_PROTOCOL.md
│   ├── THREAT_MODEL.md
│   ├── PATENT_CLAIM_SEEDS.md
│   └── hardware/INTEGRATION_SPEC.md
├── src/eczis/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── schemas.py
│   ├── camera/
│   │   ├── truth_model.py
│   │   ├── estimated_model.py
│   │   ├── operators.py
│   │   ├── psf.py
│   │   ├── cfa.py
│   │   ├── noise.py
│   │   └── calibration.py
│   ├── rig/
│   │   ├── protocol.py
│   │   ├── simulated.py
│   │   └── schedule.py
│   ├── targets/
│   │   ├── base.py
│   │   ├── microglyph.py
│   │   ├── siemens.py
│   │   ├── coin.py
│   │   ├── iris.py
│   │   └── pcb.py
│   ├── recon/
│   │   ├── register.py
│   │   ├── shift_add.py
│   │   ├── linear_operator.py
│   │   ├── map_solver.py
│   │   └── regularizers.py
│   ├── gate/
│   │   ├── residuals.py
│   │   ├── thresholds.py
│   │   └── cycle.py
│   ├── provenance/
│   │   ├── classes.py
│   │   ├── observability.py
│   │   ├── accountant.py
│   │   └── calibrate.py
│   ├── field/
│   │   ├── protocol.py
│   │   ├── tiled_raster.py
│   │   └── coordinate_field.py
│   ├── seal/
│   │   ├── canonical.py
│   │   ├── manifest.py
│   │   ├── ledger.py
│   │   ├── bundle.py
│   │   └── keys.py
│   ├── bench/
│   │   ├── datasets.py
│   │   ├── harness.py
│   │   ├── metrics.py
│   │   └── report.py
│   └── export/
│       ├── payload.py
│       └── heatmap.py
├── viewer/index.html
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── adversarial/
│   ├── benchmark/
│   └── golden/
└── tools/
    ├── claims_check.py
    ├── ip_gate.py
    └── verify_release.py
```

---

## 5. Frozen Conventions

These rules are written before algorithms and cannot change silently.

- Arrays use `(height, width[, channels])` ordering.
- Image coordinates use pixel centers; origin is top-left.
- Subpixel shift is `(dx, dy)` in low-resolution sensor pixels.
- Linear scene RGB uses `float32` in `[0, 1]` before sensor clipping.
- RAW values are linearized and black-level-subtracted before reconstruction.
- CFA pattern is explicit in every frame; no global hidden default.
- All physical quantities carry units in field names, such as `exposure_s` and `pixel_pitch_um`.
- Saturated, dead, clipped, and occluded samples carry masks and never contribute as ordinary evidence.
- Randomness uses named streams derived from a recorded root seed.
- Schema, algorithm, calibration, and bundle versions are recorded independently.

---

## 6. Core Interfaces

```python
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatImage = npt.NDArray[np.float32]


class CaptureRig(Protocol):
    def get_calibration(self) -> "Calibration": ...

    def capture_burst(
        self,
        schedule: "CaptureSchedule",
        *,
        root_seed: int,
    ) -> "CaptureSet": ...


class ForwardOperator(Protocol):
    def project(
        self,
        scene_hr: FloatImage,
        observation: "ObservationParams",
    ) -> "RawFrame": ...

    def adjoint(
        self,
        residual: FloatImage,
        observation: "ObservationParams",
    ) -> FloatImage: ...


class ReconstructionMethod(Protocol):
    def reconstruct(
        self,
        capture: "CaptureSet",
        model: ForwardOperator,
        config: "ReconConfig",
    ) -> "ReconArtifacts": ...


class ContinuousRenderer(Protocol):
    def query(self, x: float, y: float, scale: float) -> "QueryResult": ...


class InferencePrior(Protocol):
    def propose(
        self,
        reconstruction: FloatImage,
        context: "PriorContext",
    ) -> FloatImage: ...


def consistency_gate(
    proposal_hr: FloatImage,
    capture: "CaptureSet",
    model: ForwardOperator,
    thresholds: "GateThresholds",
) -> "GateResult": ...


def seal_bundle(
    capture: "CaptureSet",
    artifacts: "ReconArtifacts",
    key: "SealKey",
    output: Path,
) -> Path: ...


def verify_bundle(path: Path, key: "SealKey") -> "VerifyReport": ...
```

The release cannot proceed until interface tests freeze serialization and backward-compatibility behavior.

---

## 7. Provenance Semantics

### M — Measured

Use only where a displayed value has sufficient direct calibrated sensor support, valid masks, and an evidence footprint above the configured threshold. M does **not** mean that multiple RAW frames have low variance.

### R — Reconstructed

Use where multi-observation evidence makes the feature observable under the estimated model, the reconstruction passes the noise-normalized cycle gate, and prior contribution stays below the R limit.

### I — Inferred

Use where content is plausible and may be observation-consistent but is not uniquely determined by the available measurements, or where prior contribution exceeds the R limit.

### U — Unsupported

Use where observations are missing, contradictory, saturated, uncalibrated, outside supported scale, or inconsistent with the model.

### Monotonicity rule

A transformation may keep or demote a class, never promote it. Promotion requires new physical observations plus a new sealed capture event.

### Fundamental limitation

Cycle consistency can reject details that measurably contradict captures. It cannot prove that all observation-consistent details are factually unique. Non-identifiable alternatives must remain I, even when their residual is low.

---

## 8. Phase-by-Phase Execution

A phase is complete only when:

1. its code is merged locally;
2. `uv run mypy --strict src tests` passes;
3. `uv run ruff check .` passes;
4. its required tests pass;
5. its claim mappings pass `claims_check.py`;
6. its generated artifacts verify; and
7. the previous phase remains green.

No phase may weaken an earlier acceptance threshold without a versioned decision record.

### Phase 0 — IP Isolation and Reproducible Scaffold

**Build**

- private repository and protected default branch;
- `IP_STATUS.yaml` with `UNFILED` state;
- pre-commit/pre-push IP disclosure guard;
- Python 3.12 and `uv.lock`;
- strict typing, linting, testing, coverage, and Windows/Linux CI;
- deterministic configuration loader;
- canonical JSON encoder;
- release-claims registry.

**Acceptance**

- a public remote or missing `# UNFILED-IP` header is blocked;
- clean checkout installs only from `uv.lock`;
- same seed produces same fixture hashes on the same platform;
- cross-platform numerical tests pass within declared tolerances;
- claim registry rejects missing, stale, or unexecuted test IDs.

### Phase 1 — Schemas, Units, and Serialization

**Build**

- immutable versioned schemas for calibration, observations, RAW frames, masks, schedules, captures, reconstructions, provenance, manifests, and reports;
- array metadata validation;
- canonical serialization rules;
- schema migration rejection for unknown future major versions.

**Acceptance**

- round-trip fixtures preserve values and hashes;
- invalid units, shapes, CFA patterns, and masks fail closed;
- unknown required fields or major versions fail explicitly;
- serialized field ordering is canonical.

### Phase 2 — Independent Camera Models

**Build**

`TruthCameraModel` includes:

- spatially varying or richer PSF option;
- chromatic aberration;
- subpixel motion;
- defocus;
- CFA sampling;
- Poisson-Gaussian noise;
- quantization, black level, saturation, and nonlinearity;
- illumination variation.

`EstimatedCameraModel` supports a simpler calibrated subset used by reconstruction.

**Required operator tests**

- PSF energy conservation;
- analytic sinusoid shift tests under declared boundary conditions;
- CFA sample-location correctness;
- Poisson-Gaussian statistics;
- clipping and mask correctness;
- deterministic named random streams;
- forward/adjoint inner-product identity;
- model-mismatch scenarios are non-zero and recorded.

**Gate**

The forward/adjoint identity must pass within tolerance. Benchmark observations must not be generated by the exact object used for reconstruction.

### Phase 3 — Targets and Simulated Rig

**Build**

- deterministic microglyph target with an internal decoder;
- Siemens star;
- coin relief/scratch target;
- iris radial-fiber target;
- PCB trace/silkscreen target;
- shift, focus, exposure, and illumination schedules;
- simulated rig implementing `CaptureRig`.

OCR is a secondary metric only. The release-critical microtext metric uses known procedural glyph identities so platform OCR variance cannot determine pass/fail.

**Acceptance**

- targets contain verified features below one sensor-pixel footprint;
- capture schedules contain distinct information-bearing observations;
- duplicate or degenerate schedules are rejected;
- all captures carry calibration, masks, seeds, and hashes;
- capture replay reproduces the same same-platform artifact hashes.

### Phase 4 — Registration and Reconstruction Baselines

**Build**

1. bicubic baseline;
2. single-frame deconvolution baseline;
3. registered shift-and-add;
4. regularized MAP solver using the validated forward/adjoint operator;
5. robust frame weighting and residual diagnostics.

Use quadratic/Tikhonov regularization for the first CG implementation. TV regularization may be added only through a mathematically valid solver such as ADMM, PDHG, or documented IRLS—not by labeling ordinary CG as TV optimization.

**Acceptance**

- registration error below the declared tolerance on textured fixtures;
- low-texture registration fails closed to U;
- shift-and-add beats bicubic on pre-registered challenge cases;
- MAP beats shift-and-add by a pre-registered margin on at least three target families;
- one corrupted frame is down-weighted and logged;
- PSF mismatch increases residual alarms instead of being silently absorbed;
- noise-only input does not generate verified structure;
- solver convergence and stopping criteria are recorded.

### Phase 5 — Noise-Aware Cycle Gate

**Build**

- forward projection of candidate reconstruction;
- heteroscedastic residual normalization using predicted sensor variance;
- per-frame and per-region negative log-likelihood or normalized chi-square score;
- residual aggregation with saturation/invalid-sample masks;
- calibrated thresholds from a calibration split only;
- challenge set held out from threshold selection.

**Acceptance**

- clean reconstructed regions meet the registered false-demotion target;
- injected false features whose projected energy exceeds the detectability threshold are rejected at the registered detection rate;
- features below the physical detectability threshold are never promoted to R merely because residuals are low;
- a candidate matching all but one observation is rejected in the affected region;
- thresholds remain stable across at least three SNR regimes;
- all demotions include machine-readable reasons.

**Prohibited claim**

Do not claim 100% universal hallucination detection. The valid claim is detection performance over a defined challenge set and detectability regime.

### Phase 6 — Observability and Provenance

**Build**

- local sampling-support map;
- local observability score from the forward operator or an approximation of its local Gram/Fisher information;
- prior-contribution map;
- contradiction and invalid-sample maps;
- conservative M/R/I/U accountant;
- calibrated confidence scores.

**Acceptance**

- hand-constructed fixtures map to normative classes;
- contradictory observation subsets map to U;
- direct measurement support is not inferred from low frame variance;
- prior contribution cannot produce M or elevate I to R;
- empty evidence maps entirely to U;
- expected calibration error meets the registered synthetic target without using test data for tuning.

### Phase 7 — Continuous Rendering

**Build**

Implement two backends behind one interface:

1. `TiledRasterRenderer` — deterministic release baseline using an LOD pyramid and continuous interpolation.
2. `CoordinateFieldRenderer` — optional coordinate MLP or Fourier-feature field, not required for the first green release.

The renderer carries provenance and confidence through every query. A renderer is never permitted to elevate class.

**Acceptance**

- source-grid values reproduce within tolerance;
- arbitrary-coordinate queries have bounded continuity and seam error;
- queried class equals the conservative minimum of source evidence class, scale-support class, and renderer class;
- zoom beyond supported bandwidth demotes to I or U;
- optional coordinate-field failures cannot block the deterministic baseline release.

### Phase 8 — Evidence Manifest, Seal, and Ledger

**Build**

- canonical manifest with schema versions and sorted paths;
- SHA-256 hash for every artifact;
- manifest Merkle root or equivalent deterministic aggregate;
- HMAC-SHA-256 over the canonical manifest;
- DPAPI-protected local key on Windows;
- clearly marked development-only environment-key fallback;
- append-only event ledger with crash-safe transactions;
- same-platform replay metadata including Python, dependency lock, OS, CPU, and numerical-library identity.

HMAC proves integrity to a party holding the secret. It does not provide third-party nonrepudiation. Public-key signing and trusted timestamps are deferred.

**Acceptance**

- one-bit changes, file deletion, insertion, reorder, or manifest tampering fail verification;
- wrong key fails cleanly;
- interrupted sealing leaves no valid partial bundle;
- ledger chain deletion or reorder is detected;
- same-platform replay is bit-identical where feasible;
- cross-platform replay is assessed by declared numerical tolerances, not promised bit-for-bit.

### Phase 9 — CLI and End-to-End Workflow

**Commands**

```text
eczis doctor
eczis target create
eczis capture simulate
eczis reconstruct
eczis gate
eczis provenance
eczis render
eczis seal
eczis verify
eczis bench
eczis export
eczis demo
```

**Acceptance**

- `eczis demo` performs a complete local microglyph run from target generation through verified viewer export;
- every command has `--config`, `--output`, and deterministic seed behavior where applicable;
- invalid or incomplete inputs fail closed;
- resumption never treats an incomplete stage as completed;
- structured JSON logs contain run, artifact, and claim IDs.

### Phase 10 — Benchmark and Adversarial Validation

**Dataset partitions**

- calibration seeds — thresholds only;
- development seeds — implementation tuning;
- locked test seeds — final reporting only;
- mismatch challenge seeds — robustness;
- adversarial feature-injection seeds — gate testing.

**Methods**

- bicubic;
- single-frame deconvolution;
- registered shift-and-add;
- regularized MAP;
- full ECZIS gate plus provenance.

An AI prior may be reported as an optional experimental method but cannot define the v0.1R1 pass condition.

**Metrics**

- procedural microglyph identity accuracy;
- secondary OCR accuracy;
- edge-location RMSE;
- feature precision, recall, and F1;
- verified-feature recovery rate;
- false verified-feature rate;
- hallucinated-feature rate by class;
- false-demotion and false-promotion rates;
- residual calibration;
- expected calibration error;
- runtime, peak memory, and bundle size.

**Acceptance contract**

Thresholds are committed in `VALIDATION_PROTOCOL.md` before the locked test set is executed. Results are valid only when:

- ECZIS improves verified-feature recovery over shift-and-add on a majority of target families;
- false R/M promotion stays below the registered limit;
- feature injections above the detectability threshold meet the registered detection rate;
- mismatch challenge performance degrades transparently and shifts uncertain areas toward I/U;
- no result uses the exact truth model as its estimated reconstruction model;
- the report includes confidence intervals and test-set size.

### Phase 11 — Forensic Viewer

**Build**

- offline HTML/JavaScript viewer;
- pan and zoom;
- M/R/I/U overlay;
- confidence and residual overlays;
- region inspector with supporting observations, masks, parameters, versions, and hashes;
- explicit legend and finite-evidence disclaimer;
- payload schema validation.

**Acceptance**

- viewer opens without network access;
- bundle verification state is visible;
- no U region is displayed as sharpened verified detail in forensic mode;
- region metadata resolves to manifest hashes;
- viewer cannot alter or reseal evidence.

### Phase 12 — Physical Rig Integration Contract

**Document only in v0.1R1**

Define:

- RAW camera adapter requirements;
- XY stage precision, backlash, repeatability, and homing requirements;
- focus rail control;
- LED direction/intensity calibration;
- dark-frame, flat-field, PSF, noise, saturation, and shift-lattice calibration;
- timestamp and trigger requirements;
- microscope ground-truth protocol;
- real-capture acceptance reruns;
- replacement of synthetic-only claims with real-world claims only after successful validation.

---

## 9. Required Test Families

Exact test counts are not frozen until Phase 1 schemas and parameter matrices are frozen. Freezing an arbitrary total before parameterization encourages shallow tests.

The release must include at least these families:

| ID | Family | Minimum purpose |
|---|---|---|
| T0 | Scaffold/IP | disclosure gate, locked install, claims registry |
| T1 | Schemas | validation, canonical serialization, version rejection |
| T2 | Camera operators | physics, noise, masks, forward/adjoint identity |
| T3 | Targets/rig | sub-sensor truth, schedule validity, capture replay |
| T4 | Registration/reconstruction | recovery, convergence, corruption, mismatch |
| T5 | Cycle gate | calibrated residuals, detectable false features, K-1 attack |
| T6 | Provenance | observability, contradiction, monotonic classes, ECE |
| T7 | Rendering | continuity, LOD, class preservation, scale demotion |
| T8 | Seal/ledger | tamper, crash, wrong key, replay |
| T9 | CLI/E2E | complete deterministic workflow, resumption, failure modes |
| T10 | Bench | held-out metrics, confidence intervals, mismatch robustness |
| T11 | Viewer | offline use, schema, evidence traceability |

The final test count is generated from the frozen matrix and recorded in the release manifest. Passing percentage is 100%; skipped release tests are failures unless explicitly classified as platform-not-applicable.

---

## 10. Dependency and Tooling Contract

Use the smallest dependable stack:

- Python 3.12;
- `uv` for environments, locking, and command execution;
- NumPy;
- SciPy;
- Pillow for deterministic basic image I/O and built-in glyph support;
- scikit-image only where its behavior is pinned and covered by golden tests;
- Pydantic v2 or frozen dataclasses for versioned schemas;
- pytest, pytest-cov, Hypothesis;
- mypy strict;
- Ruff;
- optional OpenCV isolated behind an adapter, not a core requirement;
- no network dependency at runtime;
- no downloaded model weights in v0.1R1.

Every dependency is exact-locked in `uv.lock`. Numerical backend identity is recorded in bundles.

---

## 11. Initial PowerShell Bootstrap Sequence

Run only after the repository has been confirmed private.

```powershell
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = Join-Path $HOME 'source\repos\eczis'

if (Test-Path $Repo) {
    throw "Refusing to overwrite existing path: $Repo"
}

New-Item -ItemType Directory -Path $Repo -Force | Out-Null
Set-Location $Repo

git init -b main
uv python pin 3.12
uv init --package --python 3.12

uv add numpy scipy pillow pydantic
uv add --dev pytest pytest-cov hypothesis mypy ruff

New-Item -ItemType Directory -Force -Path @(
    'docs\hardware',
    'src\eczis\camera',
    'src\eczis\rig',
    'src\eczis\targets',
    'src\eczis\recon',
    'src\eczis\gate',
    'src\eczis\provenance',
    'src\eczis\field',
    'src\eczis\seal',
    'src\eczis\bench',
    'src\eczis\export',
    'tests\unit',
    'tests\integration',
    'tests\adversarial',
    'tests\benchmark',
    'tests\golden',
    'tools',
    'viewer',
    '.github\workflows'
) | Out-Null

@('# UNFILED-IP', '', 'ECZIS private development repository.') |
    Set-Content -Path README.md -Encoding utf8NoBOM

@('status: UNFILED', 'public_disclosure_allowed: false') |
    Set-Content -Path IP_STATUS.yaml -Encoding utf8NoBOM

uv lock
uv run python -c "import numpy, scipy, PIL, pydantic; print('ECZIS bootstrap OK')"
uv run ruff check .

git add .
git commit -m 'chore: initialize private ECZIS v0.1R1 scaffold'
```

This bootstrap deliberately does not create or push a GitHub repository. Remote creation occurs only after private visibility and account ownership are explicitly verified.

---

## 12. Stop Conditions

Stop the build and do not paper over the result when any of these occurs:

1. The repository is public or ownership is uncertain.
2. Truth and estimated camera models become identical in benchmark execution.
3. The forward operator lacks a passing adjoint test.
4. Thresholds are tuned using locked test results.
5. A renderer elevates provenance.
6. A low-residual but non-identifiable detail is labeled R.
7. A corrupted or partial bundle verifies.
8. A real-world claim is made from synthetic-only evidence.
9. A benchmark metric lacks a ground-truth definition.
10. A release claim has no passing test and sealed result digest.

---

## 13. Definition of Done for v0.1R1

v0.1R1 is done only when all of the following are true:

- one command runs the deterministic synthetic demonstration;
- the truth simulator and estimated reconstruction model are independently implemented;
- reconstruction improves at least one pre-registered objective recovery metric over baselines;
- the gate identifies detectable contradictions and refuses uniqueness claims for non-identifiable detail;
- provenance is conservative and monotonic through rendering;
- evidence bundles detect every tested mutation and partial-write scenario;
- benchmark thresholds were frozen before locked-test execution;
- all release tests pass;
- the report uses evidence-bounded language;
- the repository remains private; and
- no physical-rig, patent, certification, or universal-performance claim is made.

---

## 14. First Release Demonstration

The first demonstration uses the procedural microglyph target because it provides exact feature identities and objective false-feature scoring.

The demo must produce:

1. ground-truth target;
2. simulated RAW burst from `TruthCameraModel`;
3. calibration estimate for `EstimatedCameraModel`;
4. bicubic, shift-and-add, and MAP reconstructions;
5. injected false-feature challenge;
6. cycle residual maps;
7. M/R/I/U provenance map;
8. continuous zoom payload;
9. sealed evidence bundle;
10. verification report;
11. benchmark summary with confidence intervals; and
12. offline forensic viewer.

The winning demonstration is not “the sharpest-looking image.” It is a reconstruction that recovers more verified features while refusing to label unsupported detail as measured or reconstructed.
