# UNFILED-IP
"""Disclosure gate: IP_STATUS must block disclosure; every text artifact carries the header."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".ps1", ".html"}
SKIP_DIRS = {".git", ".venv", "node_modules", "artifacts", "build", "dist", "htmlcov"}


def candidate_files(root: Path) -> list[Path]:
    """Tracked + untracked-but-not-ignored files when in git; filesystem walk otherwise."""
    try:
        output = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout
        return [root / line for line in output.splitlines() if line and (root / line).is_file()]
    except (OSError, subprocess.CalledProcessError):
        return [
            p for p in root.rglob("*")
            if p.is_file() and not any(part in SKIP_DIRS or part.startswith(".") for part in p.relative_to(root).parts[:-1])
        ]


def main(root: Path = ROOT) -> int:
    status = (root / "IP_STATUS.yaml").read_text(encoding="utf-8")
    errors: list[str] = []
    if not re.search(r"(?m)^status:\s*UNFILED\s*$", status) or not re.search(
        r"(?m)^public_disclosure_allowed:\s*false\s*$", status
    ):
        errors.append("IP_STATUS.yaml does not block disclosure (need 'status: UNFILED' and 'public_disclosure_allowed: false')")
    checked = 0
    for path in candidate_files(root):
        if path.suffix.lower() not in HEADER_SUFFIXES:
            continue
        checked += 1
        first = path.read_text(encoding="utf-8", errors="ignore")[:200]
        if "UNFILED-IP" not in first:
            errors.append(f"missing UNFILED-IP header: {path.relative_to(root).as_posix()}")
    if errors:
        print("\n".join(errors))
        return 2
    print(f"IP gate PASS ({checked} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
