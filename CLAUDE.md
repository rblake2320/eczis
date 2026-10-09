# UNFILED-IP
# CLAUDE.md — Claude Code orientation for ECZIS

**Read `AGENTS.md` first — it is the canonical agent contract for this repo
(shared with Codex and all other agents).** This file adds only the pointers
Claude Code needs on this machine.

## The four rules that must never break

1. **PRIVATE repo, UNFILED IP** (`IP_STATUS.yaml`): no public disclosure of
   any kind until the owner confirms a provisional filing.
2. **`# UNFILED-IP` header** in the first 200 chars of every committed
   `.py/.md/.yaml/.yml/.ps1/.html` file — `python tools/ip_gate.py` must PASS.
3. **Claim language**: "infinitely sampleable output, finite and explicitly
   classified evidence"; synthetic software reduction-to-practice, never
   physical-camera performance. `python tools/claims_check.py` must PASS.
4. **Personal media and evidence bundles never enter git history** — see
   AGENTS.md working agreement. Real inputs have no ground truth → no R/M
   promotion, ever.

## Quick reference (this machine)

- Working copy: `D:\Projects\eczis` (`C:\Users\techai\eczis` is a junction to it; `.mcp.json` uses the D: path) · remote: `rblake2320/eczis` (private)
- Green path: `.\Run-ECZIS.ps1 -OpenViewer` (PowerShell 7)
- Gates before any push: `uv run pytest` + `uv run python tools/ip_gate.py` +
  `uv run python tools/claims_check.py`
- CI runs on `main` only (ubuntu + windows, Python 3.12 via uv).
- Python 3.12 at `C:\Python312\python.exe`; `uv` manages the venv.

## Relationship to other projects

ECZIS is a standalone personal-IP project. Do NOT cross-link it with
design-beast, PKA, or any other repo's infrastructure, and never reference
Citizant/government work here. Patent-adjacent: treat like TSK/GUMBO —
provisional before any public artifact.
