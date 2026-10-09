# UNFILED-IP
"""Blind-analysis control: freeze the grading policy BEFORE the physical truth is entered.

The fingerprint binds every file and constant that decides which points become R/I/U or get
refused (grade_evidence, grade_dense, scan3d, the benchmark's axis gates) plus the git commit.
`freeze()` writes it; `check()` says whether the policy in use is the frozen one. The phantom
benchmark records that status in every output, so a run whose policy changed after the diameter
was known is visibly unblinded. EXPERIMENTAL; never changes production M/R/I/U.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
POLICY_FILES = ("src/eczis/lab/grade_evidence.py", "src/eczis/lab/grade_dense.py", "src/eczis/lab/scan3d.py",
                "tools/phantom_benchmark.py")
DEFAULT_PATH = ROOT / "artifacts" / "benchmark" / "policy_freeze.json"


def fingerprint() -> dict[str, Any]:
    hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in POLICY_FILES if (ROOT / f).exists()}
    from eczis.lab import scan3d
    from eczis.lab.grade_evidence import POLICY

    constants = {"grade_evidence.POLICY": POLICY, "scan3d.MIN_PARALLAX_PX": scan3d.MIN_PARALLAX_PX,
                 "scan3d.MAX_GLARE_FRAC": scan3d.MAX_GLARE_FRAC, "scan3d.MAX_DARK_FRAC": scan3d.MAX_DARK_FRAC}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    digest = hashlib.sha256(json.dumps({"files": hashes, "constants": constants}, sort_keys=True, default=str).encode()).hexdigest()
    return {"files": hashes, "constants": constants, "commit": commit, "digest": digest}


def freeze(path: Path = DEFAULT_PATH, note: str = "") -> dict[str, Any]:
    rec = {**fingerprint(), "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note,
           "rule": "frozen before the tube diameter was entered or any benchmark error examined"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


def check(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    """Status of the policy in use against the freeze: BLIND (matches), CHANGED (differs), or NOT_FROZEN."""
    if not path.exists():
        return {"status": "NOT_FROZEN", "detail": f"no freeze at {path}; run tools/freeze_policy.py before entering the diameter"}
    frozen = json.loads(path.read_text(encoding="utf-8"))
    now = fingerprint()
    if frozen.get("digest") == now["digest"]:
        return {"status": "BLIND", "frozen_at": frozen.get("frozen_at"), "commit": frozen.get("commit"), "digest": now["digest"]}
    changed = [f for f in now["files"] if frozen.get("files", {}).get(f) != now["files"][f]]
    changed += [k for k in now["constants"] if frozen.get("constants", {}).get(k) != now["constants"][k]]
    return {"status": "CHANGED", "frozen_at": frozen.get("frozen_at"), "changed": changed,
            "detail": "policy differs from the freeze: this run is UNBLINDED unless re-frozen before the truth was seen"}
