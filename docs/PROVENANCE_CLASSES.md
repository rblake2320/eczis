# UNFILED-IP
# Provenance Classes

- **M — Measured:** sufficient direct calibrated support for the displayed value. The HR reconstruction currently does not label individual HR pixels M.
- **R — Reconstructed:** observation-supported, locally observable, low-prior content that passes the cycle gate.
- **I — Inferred:** plausible or observation-consistent content that is not uniquely constrained, or content failing the R criteria.
- **U — Unsupported:** missing, invalid, contradictory, saturated, or beyond supported scale.

Classes are ordered `U < I < R < M`. Rendering combines classes with a minimum operation. Promotion requires new captured evidence and a new sealed event.
