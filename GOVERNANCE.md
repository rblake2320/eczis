# UNFILED-IP
# ECZIS Governance

## Ownership

Ron Blake is the project owner and final authority for architecture, release claims, access,
licensing, patent-disclosure decisions, and acceptance of contributions.

## Decision records

Material architecture decisions must be documented in `docs/decisions/` using an ADR before
merge. A decision is material when it affects capture physics, provenance semantics, evidence
sealing, benchmark validity, hardware contracts, or patent claim posture.

## Merge policy

- `main` must remain releasable.
- Changes enter through pull requests after the initial repository bootstrap.
- CI, IP gate, claims gate, tests, and owner review are mandatory.
- Squash merge is preferred to preserve a readable development history.
- Direct pushes to `main` should be blocked after bootstrap.

## Disclosure policy

`IP_STATUS.yaml` is authoritative. While `public_disclosure_allowed: false`, the repository,
issues, discussions, artifacts, benchmarks, and derived descriptions must remain private.
