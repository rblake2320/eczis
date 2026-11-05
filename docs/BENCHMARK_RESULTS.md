# UNFILED-IP
# ECZIS Benchmark Results — locked_test split

> Infinitely sampleable output, finite and explicitly classified evidence.
> Synthetic software reduction-to-practice only. No physical-camera performance is implied.

- Command: `uv run eczis bench --split locked_test` (default `DemoConfig`, thresholds per `VALIDATION_PROTOCOL.md`)
- Seeds [701, 809, 907] x 5 families = 15 cases per arm; arms: `nominal` (truth PSF 0.82 LR px) and `psf_mismatch` (truth PSF 1.0 LR px); estimated PSF 0.70 in both.
- Cells: mean [95% percentile-bootstrap CI over 3 seeds]. With n = 3 the interval shows dispersion, not precision.
- Environment: Python 3.12.3, numpy 2.5.1, scipy 1.18.0, Linux x86_64, eczis 0.1.0.

## Nominal arm — reconstruction PSNR (dB)

| Family | MAP | Shift-add | 1-frame deconv | Bicubic |
|---|---:|---:|---:|---:|
| coin | 24.37 [24.30, 24.42] | 21.93 [21.92, 21.93] | 22.74 [22.56, 22.86] | 21.79 [21.72, 21.85] |
| iris | 19.21 [19.18, 19.24] | 19.09 [19.08, 19.10] | 18.99 [18.98, 19.00] | 19.02 [19.00, 19.03] |
| microglyph | 11.87 [11.86, 11.87] | 11.35 [11.35, 11.35] | 11.58 [11.57, 11.59] | 11.38 [11.37, 11.38] |
| pcb | 14.89 [14.88, 14.91] | 13.49 [13.49, 13.49] | 14.17 [14.17, 14.17] | 13.60 [13.60, 13.61] |
| siemens | 14.66 [14.66, 14.68] | 13.57 [13.57, 13.57] | 14.90 [14.87, 14.93] | 13.95 [13.93, 13.97] |

## Nominal arm — evidence

| Family | R frac | False-R (band) | False-R (full band) | ECE (raw) | Gate demotion | Injected R/M |
|---|---:|---:|---:|---:|---:|---:|
| coin | 0.245 [0.235, 0.257] | 0.000 [0.000, 0.000] | 0.012 [0.008, 0.016] | 0.380 [0.376, 0.388] | 0.859 [0.849, 0.870] | 0.000 [0.000, 0.000] |
| iris | 0.343 [0.317, 0.359] | 0.000 [0.000, 0.000] | 0.026 [0.021, 0.034] | 0.378 [0.370, 0.390] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| microglyph | 0.002 [0.002, 0.003] | 0.119 [0.000, 0.217] | 0.688 [0.272, 0.913] | 0.349 [0.342, 0.357] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| pcb | 0.006 [0.001, 0.012] | 0.065 [0.000, 0.172] | 0.375 [0.172, 0.510] | 0.427 [0.420, 0.432] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| siemens | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.209 [0.202, 0.216] | 0.911 [0.906, 0.917] | 0.000 [0.000, 0.000] |

## PSF-mismatch arm — reconstruction PSNR (dB)

| Family | MAP | Shift-add | 1-frame deconv | Bicubic |
|---|---:|---:|---:|---:|
| coin | 22.61 [22.61, 22.62] | 21.43 [21.43, 21.43] | 21.64 [21.57, 21.69] | 21.23 [21.22, 21.25] |
| iris | 19.00 [18.99, 19.02] | 19.01 [19.00, 19.01] | 18.86 [18.85, 18.88] | 18.92 [18.92, 18.93] |
| microglyph | 11.49 [11.48, 11.50] | 11.28 [11.28, 11.28] | 11.38 [11.37, 11.38] | 11.29 [11.29, 11.29] |
| pcb | 14.00 [13.99, 14.02] | 13.25 [13.24, 13.25] | 13.71 [13.70, 13.72] | 13.33 [13.32, 13.33] |
| siemens | 13.69 [13.67, 13.71] | 13.03 [13.03, 13.03] | 14.00 [13.98, 14.01] | 13.30 [13.29, 13.31] |

## PSF-mismatch arm — evidence

| Family | R frac | False-R (band) | False-R (full band) | ECE (raw) | Gate demotion | Injected R/M |
|---|---:|---:|---:|---:|---:|---:|
| coin | 0.255 [0.250, 0.259] | 0.011 [0.007, 0.017] | 0.024 [0.020, 0.030] | 0.189 [0.188, 0.190] | 0.866 [0.865, 0.870] | 0.000 [0.000, 0.000] |
| iris | 0.356 [0.335, 0.378] | 0.000 [0.000, 0.000] | 0.029 [0.015, 0.049] | 0.367 [0.358, 0.374] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| microglyph | 0.034 [0.025, 0.040] | 0.475 [0.443, 0.535] | 0.951 [0.903, 0.987] | 0.028 [0.024, 0.032] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| pcb | 0.053 [0.048, 0.062] | 0.205 [0.179, 0.230] | 0.761 [0.714, 0.801] | 0.166 [0.160, 0.169] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| siemens | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.133 [0.130, 0.135] | 0.905 [0.901, 0.906] | 0.000 [0.000, 0.000] |

## Acceptance (as registered)

| Criterion | Result |
|---|---|
| MAP beats shift-and-add (mean PSNR) on >= 3 families | **5 / 5** |
| Injected pixels labeled R or M, any case | **0** (30 cases) |
| M emitted for HR reconstructions | **0** |
| Coin / iris false-R (supported band) <= 0.02, both arms | **holds** (max 0.011) |

## Findings that are NOT claims (open defects / limits)

1. **Calibration-uncertainty gap (open, P1).** On microglyph and PCB, the few R pixels
   (0.2 % / 0.6 % of the image) have supported-band false-R of 12 % / 6.5 %, rising to
   **47 % / 21 % under PSF mismatch**, where R grows to 3.4 % / 5.3 %. Root cause: the
   identifiability band is computed from the *estimated* PSF; when the true PSF is wider,
   the band is optimistic, recon content looks in-band, and truth detail the operator
   cannot actually transmit is scored as observable. Fix path: compute the band from a
   pessimistic PSF bound (estimate plus calibrated uncertainty), and/or demote R where the
   gate's residual rises with PSF-perturbation sensitivity. Until fixed, R on dense
   sub-resolution targets under uncalibrated PSF error must not be relied upon.
2. **Full-band false-R is large on HF-dense targets** (microglyph 0.69, PCB 0.38): truth
   detail beyond the band is unobservable by construction. This is why R is defined as a
   supported-band claim (ADR-0002).
3. **Uncalibrated confidence is poorly calibrated** (ECE 0.21–0.43 nominal). The
   calibration-split calibrator reaches held-out ECE <= 0.10 on the registered iris case
   (C13), but it is not yet applied inside the pipeline.
4. **Gate demotion rate for the injected H is below 0.95 on coin (0.86) and Siemens (0.91)**,
   where clipping at full scale reduces the injected energy. End-to-end injected R/M is still 0.
5. **Baseline ordering is not uniform:** shift-and-add trails bicubic on microglyph and
   Siemens; single-frame deconvolution beats MAP on Siemens (14.90 vs 14.66 dB).
