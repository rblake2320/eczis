# UNFILED-IP
from __future__ import annotations

import re
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    registry = (root / "RELEASE_CLAIMS.yaml").read_text(encoding="utf-8")
    required = set()
    for block in re.findall(r"tests:\s*\[([^\]]+)\]", registry):
        required.update(item.strip() for item in block.split(",") if item.strip())
    test_text = "\n".join(path.read_text(encoding="utf-8") for path in (root / "tests").rglob("test_*.py"))
    missing = sorted(name for name in required if f"def {name}(" not in test_text)
    if missing:
        print("Missing claim tests:", ", ".join(missing))
        return 2
    print(f"Claims check PASS: {len(required)} mapped tests found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
