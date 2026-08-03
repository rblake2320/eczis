# UNFILED-IP
# Architecture

```text
Procedural target
      │
      ▼
TruthCameraModel ──► RAW-like burst + masks + schedule
                         │
                         ▼
EstimatedCameraModel ─► bicubic / shift-add / MAP
                         │
                         ▼
Noise-aware reprojection gate
                         │
                         ▼
Observability + M/R/I/U accountant
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
Continuous renderer            Manifest / ledger / HMAC
            │                         │
            └────────────┬────────────┘
                         ▼
                  Offline viewer
```

The truth and estimated models are independent by construction. The truth path includes reflect-boundary cubic motion, spatially varying blur, nonlinear response, and Poisson-Gaussian noise. The estimated path is a periodic linear operator with an exact adjoint for MAP reconstruction and cycle scoring.
