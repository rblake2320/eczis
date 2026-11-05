# UNFILED-IP
# ECZIS Development Roadmap

## R1 — Repository and reproducibility (done in v0.1.1)

- Cross-platform GitHub Actions on Windows and Ubuntu.
- Locked Python environment and deterministic demo replay.
- Coverage reporting, claims gate, and IP-disclosure gate.
- Versioned schemas and migration policy.

## R2 — Validation expansion (partially done in v0.1.1)

Done: five-family matrix, splits + leakage guard, observability/identifiability for R, K-1 gate, mismatch arm, ECE + calibrator. Open: SNR-stratified calibration curves, larger seed sets for tighter intervals.


- Run all five target families through the locked benchmark matrix.
- Add calibration, development, locked-test, mismatch, and adversarial dataset splits.
- Replace single-threshold reporting with SNR-stratified calibration curves.
- Add observability and local-condition diagnostics for R eligibility.
- Expand false-detail challenges beyond one microglyph family.

## R3 — Continuous field (backend landed in v0.1.1; GPU profiling open)

- Implement the optional coordinate MLP/Fourier-feature backend.
- Preserve provenance and confidence exactly through field encoding.
- Add seam, scale-transition, and evidence-boundary tests.
- Benchmark CPU and RTX 5090 execution profiles.

## R4 — Evidence security (DPAPI + anchored ledger done in v0.1.1; signatures/timestamps open)

- Add Windows DPAPI/CNG key protection.
- Introduce asymmetric signatures and signed release manifests.
- Add trusted timestamp integration behind a versioned interface.
- Define evidence retention, export, and independent-verification procedures.

## R5 — Physical rig adapter

- Implement the versioned `CaptureRig` hardware contract.
- Add camera, XY stage, focus rail, and programmable illumination adapters.
- Calibrate PSF, shifts, radiometry, saturation, and timing.
- Re-run synthetic claims against real captures before making optical-performance claims.

## R6 — Productization research

- Hardware bill of materials and controlled-object scanner enclosure.
- Operator workflow and automated capture planning.
- Artifact database, comparison history, and evidence audit trail.
- Candidate markets: collectibles, artwork, electronics inspection, and controlled metrology.

No roadmap item authorizes public disclosure or expands a release claim without passing evidence.
