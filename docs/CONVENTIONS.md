# UNFILED-IP
# Conventions

- Arrays: `(height, width)`.
- Coordinates: top-left origin, pixel-center interpretation.
- Shift: `(dx_lr, dy_lr)` in low-resolution sensor pixels.
- Scene and RAW: linear `float32` in `[0,1]`.
- Every frame carries explicit masks and observation parameters.
- All demonstration randomness derives from a recorded root seed.
