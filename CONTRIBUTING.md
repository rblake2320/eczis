# UNFILED-IP
# Contributing to ECZIS

ECZIS is currently private, proprietary, and associated with unfiled intellectual property.
Do not copy code, screenshots, benchmark outputs, architecture diagrams, claim language, or
results into public systems without written authorization from Ron Blake.

## Required workflow

1. Create an issue describing the change and its evidence boundary.
2. Create a branch using `feature/<issue>-<name>`, `fix/<issue>-<name>`, or `research/<issue>-<name>`.
3. Add or update tests before changing a release claim.
4. Run the complete local gate:

```powershell
uv sync --all-groups
uv run ruff check .
uv run mypy src tools
uv run python tools/ip_gate.py
uv run python tools/claims_check.py
uv run pytest --cov=eczis --cov-report=term-missing --cov-fail-under=70
```

5. Open a pull request using the repository template.
6. Do not merge while CI, the IP gate, the claims gate, or review is incomplete.

## Engineering constraints

- Preserve independent truth and estimated camera models.
- Do not validate a reconstruction only with the exact model that generated its observations.
- Never promote evidence class through inference. New physical observations are required.
- Use the phrase **infinitely sampleable output, finite evidence**.
- Record random seeds and material parameters in evidence artifacts.
- Any degraded mode must be explicit and must lower or preserve—not raise—confidence.
- Never commit capture keys, `.eczis` bundles, RAW observations, generated artifacts, or secrets.

## Claim changes

Every externally meaningful claim must be mapped in `RELEASE_CLAIMS.yaml` to one or more
specific automated tests. `tools/claims_check.py` blocks orphan claims.
