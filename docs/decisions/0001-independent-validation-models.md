# UNFILED-IP
# ADR-0001: Separate Truth and Estimated Camera Models

- **Status:** Accepted
- **Date:** 2026-08-03

## Context

Using the exact forward model both to generate synthetic captures and to validate reconstruction
creates circular evidence and can overstate performance.

## Decision

ECZIS maintains independent `TruthCameraModel` and `EstimatedCameraModel` implementations. Locked
benchmarks must include intentional PSF, boundary, illumination, nonlinearity, and noise mismatch.

## Consequences

The reconstruction is evaluated under model error rather than only under self-consistency. Any
future optimization that collapses these models requires a replacement independent validator and
owner approval.
