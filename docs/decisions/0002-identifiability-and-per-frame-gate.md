# UNFILED-IP
# ADR-0002: Supported-Band Identifiability for R, and a Per-Frame Contradiction Gate

- **Status:** Accepted
- **Date:** 2026-11-05
- **Supersedes:** nothing; tightens v0.1.0 provenance semantics (no threshold was weakened)

## Context

Measured on `main@1ab8637` (demo config, microglyph target):

- 99.87 % of HR pixels were labeled **R** while MAP PSNR was 11.9 dB.
- Diagnosis: MAP error against the *band-limited* truth was 0.045 RMS; the large
  error was entirely detail above the estimated operator's transfer cutoff
  (|H(f)| < 0.1). That detail is in the near-null-space: no observation can
  distinguish it, so a low cycle residual says nothing about it.
- This is EXECUTION_PLAN stop condition 6 ("a low-residual but non-identifiable
  detail is labeled R").
- Separately, the gate averaged the normalized squared residual over all frames.
  A single frame contradicting by 6 sigma (z^2 = 36) is diluted to roughly 36/9 + 1 = 5 with 9 frames,
  below the 8.0 region threshold -- the K-1 attack from Phase 5 acceptance passed.

## Decision

1. **Identifiability is required for R.** `classify_provenance` now requires the
   proposal and estimated model. A pixel is R only if, in addition to the
   previous criteria, the local RMS of the proposal's out-of-band component
   (frequencies with |H(f)| < `min_transfer` = 0.10, Gaussian window sigma 4 HR px)
   is <= `max_unsupported_rms` = 0.02.
2. **R asserts the supported-band component only.** The report and manifest
   record the evidence cutoff (cycles / HR pixel). Benchmarks score false-R
   against band-limited truth *and* report full-band false-R alongside it, so
   unobservable truth detail is disclosed, never hidden.
3. **Per-frame contradiction criterion.** The gate also fails a region when any
   single frame's 3x3-LR local mean z^2 exceeds `max_frame_z2_threshold` = 24.
   Registered from clean smooth-target runs at 250 / 700 / 2500 e-: per-frame
   p99.9 <= 4.5, max <= 5.7. `None` disables it for ablation only.

## Consequences

- Demo R fraction drops from 99.9 % to about 1 % (microglyph strokes are beyond the
  operator's band). Smooth targets (coin, iris) keep about 25-39 % R with false-R
  below 1.2 %. This is the honest result; the earlier figure overstated evidence.
- The microglyph challenge-demotion test is unchanged and still passes; a new
  assertion requires zero injected pixels labeled R/M.
- The gate's detection *rate* for the injected H falls at low SNR (coin, 250 e-:
  0.32). It is reported, not claimed; end-to-end, injected pixels labeled R/M stay at 0
  in every regime because identifiability catches what the gate cannot.
- Fundamental limit (unchanged): no software criterion can prove that the truth
  has no unobservable detail. R is therefore a supported-band claim by definition.
