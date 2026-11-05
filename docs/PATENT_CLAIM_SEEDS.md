# UNFILED-IP
# ECZIS Provisional Claim Seeds — CONFIDENTIAL, INTERNAL ONLY

> **Do not disclose.** `IP_STATUS.yaml` is UNFILED. This file lives only in the
> private repository and is input for counsel when drafting the provisional. It is
> not a public disclosure and must not be copied into any public artifact, email,
> benchmark post, or submission until the owner confirms filing.
>
> Evidence level for every seed: **synthetic software reduction-to-practice**
> (`RELEASE_CLAIMS.yaml`, sealed demo bundle, `release_manifest.json`). No physical-camera
> performance is asserted.

## Seed 1 — Evidence-classified continuous zoom (system)

A computer-implemented imaging system comprising: a capture interface receiving a
plurality of phase-diverse sensor observations of a scene; a reconstruction engine
producing a candidate high-resolution field; a consistency gate that re-projects the
candidate through an estimated forward model that is implemented independently of the
model that produced the observations; and a provenance accountant that assigns each
output location a class from an ordered set {Unsupported < Inferred < Reconstructed <
Measured}, wherein a continuous renderer answers arbitrary-coordinate, arbitrary-scale
queries and returns the minimum of the source class, a scale-support class, and a
renderer class, such that no rendering operation can raise a class.

*Reduction to practice:* `src/eczis/provenance/accountant.py`, `src/eczis/field/*`;
tests C4.

## Seed 2 — Supported-band identifiability gate (method)

A method of labeling reconstructed image content, comprising: computing the transfer
magnitude of an estimated forward operator; partitioning spatial frequencies into a
supported band (transfer >= a registered minimum) and an unsupported band; computing,
for each location, a local energy of the candidate's unsupported-band component; and
withholding the Reconstructed class wherever that local energy exceeds a registered
limit, independent of the cycle residual, whereby observation-consistent but
non-identifiable detail is labeled Inferred; and recording the supported-band cutoff in
the sealed evidence manifest so that the Reconstructed label is defined as a claim about
the supported-band component only.

*Dependent seeds:* (a) scoring false-Reconstructed rates against band-limited ground
truth while also reporting full-band rates; (b) reduced confidence for non-identifiable
locations; (c) combination with a prior-contribution limit so a learned prior cannot
produce the Reconstructed class.

*Reduction to practice:* `src/eczis/provenance/observability.py`; tests C7, C8; ADR-0002.

## Seed 3 — Dual-statistic cycle gate resistant to single-observation contradiction

A consistency gate computing, from heteroscedastic Poisson-Gaussian predicted variance,
(i) a region-aggregated mean normalized squared residual across all valid observations
and (ii) a per-observation locally averaged normalized squared residual, and rejecting a
region when either statistic exceeds its registered threshold, the per-observation
threshold being registered from clean-data upper quantiles across multiple SNR regimes,
and recording which observations contradicted, so that a candidate matching all but one
observation cannot pass by dilution.

*Reduction to practice:* `src/eczis/gate/*`; tests C6 (including the ablation showing the
mean-only gate passes the attack).

## Seed 4 — Tamper-evident evidence bundle with anchored ledger (system)

An evidence package comprising: a canonical manifest listing every artifact with a
content digest and size in sorted unique order, plus replay-environment metadata; a keyed
MAC over the canonical manifest; a hash-chained append-only event ledger written under a
cross-process lock with torn-record detection and quarantine; and a keyed anchor sealing
the ledger head (count and entry hash), wherein verification rejects non-canonical
manifests even when correctly keyed, rejects unsafe archive member paths, and detects tail
truncation and wholesale chain rewrites that a bare hash chain cannot detect; and wherein
sealing builds the archive in a temporary file moved into place only after completion so
that an interrupted seal leaves no verifiable partial bundle.

*Reduction to practice:* `src/eczis/seal/*`, `src/eczis/atomic.py`; tests C3, C10, C11.

## Seed 5 — Claim-to-execution release gate (method)

A release-gating method in which each public-facing performance claim is registered with
one or more named tests, a test runner emits a machine-readable result report, and a gate
refuses release unless every mapped test executed and passed (skips allowed only when
explicitly classified platform-not-applicable), emitting a digest-bound claims-evidence
record and an optionally keyed release manifest that freezes the test count.

*Reduction to practice:* `tools/claims_check.py`, `tools/verify_release.py`.

## Evidence to preserve with the filing

- Commit SHA of this branch and the CI run IDs (Ubuntu + Windows).
- `release_manifest.json`, `claims_evidence.json`, `test-results.xml` CI artifacts.
- A sealed demo bundle (`eczis_demo.eczis`) plus `verification.json` — stored **outside** git.
- `docs/BENCHMARK_RESULTS.md` and the locked-test `benchmark_matrix.json`.
