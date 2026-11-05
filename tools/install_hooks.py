# UNFILED-IP
"""Point git at tools/hooks so pre-commit/pre-push gates run locally."""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    hooks = ROOT / "tools" / "hooks"
    for hook in hooks.iterdir():
        if hook.is_file():
            hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    subprocess.run(["git", "config", "core.hooksPath", "tools/hooks"], check=True, cwd=ROOT)
    print("Installed: git core.hooksPath -> tools/hooks (pre-commit, pre-push)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
