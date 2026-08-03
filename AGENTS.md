# UNFILED-IP
# AGENTS.md — orientation for AI agents (Codex, Claude, mesh peers)

Any agent working in this repository follows this file. Claude Code also reads
`CLAUDE.md` (same rules).

## Hard boundaries (non-negotiable)

1. **UNFILED IP.** `IP_STATUS.yaml` blocks public disclosure. This repo stays
   PRIVATE. No public repos, gists, blogs, benchmark posts, or outreach derived
   from this code until the owner confirms a provisional filing. Never flip
   visibility, never copy content into public repos.
2. **Every committed `.py/.md/.yaml/.yml/.ps1/.html` file starts with an
   `# UNFILED-IP` header** (first 200 chars). `python tools/ip_gate.py` enforces
   this and must PASS before any push.
3. **Claim discipline.** The mandatory phrasing is: *infinitely sampleable
   output, finite and explicitly classified evidence*. This is a **synthetic
   software reduction-to-practice** — never describe results as physical-camera
   performance. Claims live in `RELEASE_CLAIMS.yaml`; every claim maps to named
   tests, enforced by `python tools/claims_check.py`.
4. **No Citizant/government material** may enter this repo. It is personal IP
   of Ron Blake.

## What this is

ECZIS v0.1R1 — Evidence-Aware Continuous Zoom Imaging System. A deterministic,
typed, test-gated vertical slice: synthetic ground-truth targets → simulated
noisy RAW bursts (`TruthCameraModel`) → reconstruction with a deliberately
imperfect `EstimatedCameraModel` → noise-normalized cycle-consistency gate →
conservative M/R/I/U provenance → hash-chained ledger + HMAC-sealed evidence
bundle → offline forensic viewer. The *gate* is the invention; priors are
swappable.

## Commands (Windows-first, PowerShell 7, Python 3.12, uv)

```powershell
.\Run-ECZIS.ps1 -OpenViewer     # full green path: env, tests, bundle, verify, viewer
uv sync --python 3.12
uv run eczis doctor
uv run pytest                    # unit + integration + adversarial
uv run python tools/ip_gate.py       # disclosure-header gate — run before push
uv run python tools/claims_check.py  # claim→test mapping gate
```

CI (`.github/workflows/ci.yml`) runs on `main` pushes/PRs, ubuntu + windows.

## Conventions that will bite you (`docs/CONVENTIONS.md`)

- Arrays `(height, width)`; top-left origin, pixel-center coordinates.
- Shifts are `(dx_lr, dy_lr)` in LOW-RES sensor pixels.
- Scene/RAW are linear `float32` in `[0,1]`.
- All demonstration randomness derives from one recorded root seed —
  determinism is a stated quality bar, don't introduce unseeded randomness.
- Quality bar: `mypy --strict`-clean, typed, CRLF-safe, no GPU required.

## Key docs

`docs/EXECUTION_PLAN.md` (acceptance contract) · `docs/ARCHITECTURE.md` ·
`docs/THREAT_MODEL.md` · `docs/VALIDATION_PROTOCOL.md` ·
`docs/PROVENANCE_CLASSES.md` (M/R/I/U meanings) · `docs/DATA_CONTRACTS.md`.

## Working agreement

- Branch → PR into `main`; never force-push `main`.
- Run the full gate set (pytest, ip_gate, claims_check) before pushing.
- Adversarial tests (`tests/adversarial/`) are the honesty layer — a change
  that weakens the false-microglyph demotion test is wrong by definition.
- State evidence levels honestly: synthetic proof ≠ physical validation.
- **Personal media stays out of version control.** Real-photo stress sets
  (e.g. the v0.2 Private Legacy Stress Set — family photos, minors) and
  generated evidence bundles are NEVER committed, even though the repo is
  private: the repo may be published after filing; git history is forever.
  Keep them in a gitignored `evidence/`-style path or outside the tree. The
  publish script already refuses to run with evidence files present — do not
  weaken that guard. Real-input runs carry `no ground truth` and must show
  `no R/M promotion`; a change that promotes real-photo content to R/M
  without a calibrated ground-truth path is wrong by definition.
