# UNFILED-IP
# Provenance Classes

> Infinitely sampleable output, finite and explicitly classified evidence.

Classes are ordered `U < I < R < M`. Rendering combines classes with a minimum
operation. Promotion requires new captured evidence and a new sealed event.

- **M — Measured:** sufficient direct calibrated support for the displayed value.
  HR reconstructions never emit M: reconstructed HR pixels are not direct samples.
- **R — Reconstructed:** all of the following hold at the pixel:
  1. the region passes the noise-normalized cycle gate: region mean z^2 <= 8.0 **and**
     no single frame's local z^2 > 24 (the K-1 criterion, ADR-0002);
  2. at least `min_valid_frames` valid samples and sampling observability >= 0.42;
  3. prior contribution <= 0.05;
  4. **identifiable**: the proposal's local out-of-band RMS (|H(f)| < 0.10) <= 0.02.

  R asserts the **supported-band component** of the value. Detail above the
  recorded evidence cutoff is unobservable by construction and is never part of
  an R claim (ADR-0002).
- **I — Inferred:** plausible or observation-consistent content that is not
  uniquely constrained: gate failure, out-of-band detail, prior above the R
  limit, or weak observability.
- **U — Unsupported:** no valid samples (missing, saturated, invalid), or beyond
  the supported render scale (> 3x native).

## Rendering

Both renderers (`TiledRasterRenderer`, release baseline; `CoordinateFieldRenderer`,
optional) return `min(source class, scale class, backend class)`. Scale > 1.5x
native caps at I; > 3x native gives U. The coordinate field additionally caps
tiles whose fit RMSE > 0.02 at I. Property-based tests assert neither backend
ever promotes.
