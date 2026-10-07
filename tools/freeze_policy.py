# UNFILED-IP
"""Freeze (or check) the grading policy for the blind-analysis control.

    uv run python tools/freeze_policy.py                 # write artifacts/benchmark/policy_freeze.json (refuses to overwrite)
    uv run python tools/freeze_policy.py --check         # BLIND / CHANGED / NOT_FROZEN
    uv run python tools/freeze_policy.py --force --note "re-frozen before tube 2"

Run it BEFORE the tube diameter is measured/entered and before any benchmark error is looked at.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    from eczis.lab.policy_freeze import DEFAULT_PATH, check, freeze

    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing freeze (say why in --note)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.check:
        st = check(a.path)
        print(json.dumps(st, indent=2))
        return 0 if st["status"] == "BLIND" else 1
    if a.path.exists() and not a.force:
        print(f"{a.path} exists; a freeze is not silently replaced. Use --check, or --force with a --note.", file=sys.stderr)
        return 2
    rec = freeze(a.path, a.note)
    print(json.dumps({"frozen_at": rec["frozen_at"], "commit": rec["commit"], "digest": rec["digest"], "path": str(a.path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
