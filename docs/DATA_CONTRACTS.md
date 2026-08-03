# UNFILED-IP
# Data Contracts

The frozen Python dataclasses are in `src/eczis/schemas.py`. Staged captures use a compressed NPZ with three arrays (`data`, `valid`, `saturation`) and canonical JSON metadata containing calibration and observation parameters. Sealed evidence uses a canonical JSON manifest, per-file SHA-256 hashes, and HMAC-SHA-256.
