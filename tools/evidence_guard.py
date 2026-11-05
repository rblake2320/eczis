# UNFILED-IP
"""Refuse to let personal media, evidence bundles, or key material into git.

Checks every path git tracks (or is about to commit with ``--staged``).
Real-photo stress sets and generated evidence never enter history: the repo
may be published after filing and git history is forever.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import PurePosixPath

FORBIDDEN_SUFFIXES = {
    ".eczis", ".key", ".pem", ".npy", ".npz",
    ".jpg", ".jpeg", ".heic", ".heif", ".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2",
    ".tif", ".tiff", ".mov", ".mp4", ".avi",
}
FORBIDDEN_DIRS = {"artifacts", "evidence", "private", "personal", "stress_sets"}
FORBIDDEN_NAMES = {".env", "DEMO_ONLY.key", "seal.hmac"}
# Repository-owned binary assets that are allowed despite their suffix.
ALLOWED_PREFIXES = ("docs/assets/",)


def violations(paths: list[str]) -> list[str]:
    found: list[str] = []
    for raw in paths:
        path = PurePosixPath(raw)
        if raw.startswith(ALLOWED_PREFIXES):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            found.append(f"{raw}: forbidden file type {path.suffix.lower()}")
        elif path.name in FORBIDDEN_NAMES or (path.name.startswith(".env") and path.name != ".env.example"):
            found.append(f"{raw}: secret-like file")
        elif any(part.lower() in FORBIDDEN_DIRS for part in path.parts[:-1]):
            found.append(f"{raw}: inside an evidence/personal-media directory")
    return found


def git_paths(staged: bool) -> list[str]:
    command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged else ["git", "ls-files"]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return [line for line in output.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="check the staged change set only")
    parser.add_argument("paths", nargs="*", help="explicit paths (default: ask git)")
    args = parser.parse_args(argv)
    paths = args.paths or git_paths(args.staged)
    found = violations(paths)
    if found:
        print("Evidence guard FAIL -- these must never enter git history:")
        print("\n".join(f"  - {item}" for item in found))
        return 2
    print(f"Evidence guard PASS: {len(paths)} paths checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
