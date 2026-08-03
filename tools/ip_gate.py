# UNFILED-IP
from __future__ import annotations

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    status = (root / "IP_STATUS.yaml").read_text(encoding="utf-8")
    errors: list[str] = []
    if "status: UNFILED" not in status or "public_disclosure_allowed: false" not in status:
        errors.append("IP_STATUS.yaml does not block disclosure")
    for path in root.rglob("*"):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".ps1", ".html"}:
            continue
        first = path.read_text(encoding="utf-8", errors="ignore")[:200]
        if "UNFILED-IP" not in first:
            errors.append(f"missing UNFILED-IP header: {path.relative_to(root)}")
    if errors:
        print("\n".join(errors))
        return 2
    print("IP gate PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
