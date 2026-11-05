# UNFILED-IP
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from eczis.seal.environment import replay_environment

MANIFEST_SCHEMA_VERSION = "1.1"
RESERVED_NAMES = frozenset({"manifest.json", "seal.hmac"})
CLAIM_LANGUAGE = "Infinitely sampleable output, finite and explicitly classified evidence."


def aggregate_digest(digests: list[str]) -> str:
    return hashlib.sha256("".join(digests).encode("ascii")).hexdigest()


def build_manifest(root: Path, *, environment: dict[str, Any] | None = None) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in RESERVED_NAMES or relative.endswith(".eczis") or relative.endswith(".lock"):
            continue
        data = path.read_bytes()
        files.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "algorithm": "HMAC-SHA-256",
        "claim_language": CLAIM_LANGUAGE,
        "files": files,
        "aggregate_sha256": aggregate_digest([item["sha256"] for item in files]),
        "environment": replay_environment() if environment is None else environment,
    }
