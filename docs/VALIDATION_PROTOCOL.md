# UNFILED-IP
# Synthetic Validation Protocol

> Infinitely sampleable output, finite and explicitly classified evidence.
> All results are synthetic software reduction-to-practice, not physical-camera performance.

## Registered thresholds (frozen before locked-test execution)

| Parameter | Value | Source |
|---|---:|---|
| Region gate: mean normalized squared residual | 8.0 | v0.1.0 |
| Per-frame gate: 3x3-LR local z^2 (K-1 criterion) | 24.0 | ADR-0002 |
| Identifiability: min transfer \|H(f)\| | 0.10 | ADR-0002 |
| Identifiability: max out-of-band local RMS | 0.02 | ADR-0002 |
| R observability minimum / prior maximum | 0.42 / 0.05 | v0.1.0 |
| False-R ground truth: local RMS error vs band-limited truth (sigma 2 px) | > 0.08 | ADR-0002 |
| False-R registered limit (coin, three SNR regimes) | <= 0.02 | C9 |
| Microglyph challenge demotion rate (demo config) | >= 0.95 | v0.1.0 |
| Injected pixels labeled R or M (every family, every regime) | 0 | C2 |
| Held-out ECE after calibration-split calibrator (iris) | <= 0.10 | C13 |

## Development configuration

- 16 phase-diverse observations; 4x HR truth relative to the sensor grid.
- Truth PSF sigma 0.82 LR px; estimated PSF sigma 0.70 LR px (non-zero mismatch),
  plus truth-only illumination, boundary, spatially varying blur, and
  nonlinearity mismatch.
- Poisson-Gaussian truth noise (700 e- full well, 0.004 read noise).
- False H-shaped microglyph injected at amplitude 0.72.
- Adversarial and unit suites use the same physics on a 24x32 sensor with 9
  frames and 12 solver iterations (`tests/conftest.py::SMALL`).

## Dataset partitions (`src/eczis/bench/datasets.py`)

| Split | Seeds | Use |
|---|---|---|
| calibration | 101, 211, 307 | threshold / calibrator fitting only |
| development | 401, 503, 601 | implementation tuning |
| locked_test | 701, 809, 907 | final reporting only |
| mismatch | 1009, 1103, 1201 | robustness |
| adversarial | 1301, 1409, 1511 | gate attacks |

`require_split` makes leakage a hard error (`SplitLeakageError`).

## Benchmark matrix

`uv run eczis bench --split locked_test` runs 5 families x 3 seeds for the
nominal arm and a PSF-mismatch arm (truth sigma 1.0), reporting per-family means
with 95 % percentile-bootstrap intervals (2000 resamples, seed 0) and
the test-set size. With 3 seeds per family the intervals describe dispersion,
not precision. Results: `docs/BENCHMARK_RESULTS.md`.

## Known limits (reported, not claimed)

- The gate's detection *rate* for injected detail falls at low SNR (coin, 250 e-:
  0.32). End-to-end false R/M for injected pixels remains 0.
- Shift-and-add does not beat bicubic on every family; see the benchmark results.
- Cross-platform numerics are asserted within the declared tolerances in
  `tests/golden/test_golden.py` (PSNR +/- 0.05 dB, fractions +/- 0.01, target
  statistics +/- 1e-5). Serialization (schedule, manifest) is asserted exactly.
