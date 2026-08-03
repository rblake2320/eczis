# UNFILED-IP
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def build_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"manifest.json", "seal.hmac"} or relative.endswith(".eczis"):
            continue
        data = path.read_bytes()
        files.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    aggregate = hashlib.sha256("".join(item["sha256"] for item in files).encode("ascii")).hexdigest()
    return {
        "schema_version": "1.0",
        "algorithm": "HMAC-SHA-256",
        "claim_language": "Infinitely sampleable output, finite and explicitly classified evidence.",
        "files": files,
        "aggregate_sha256": aggregate,
    }
