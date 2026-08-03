# UNFILED-IP
from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import zipfile
from pathlib import Path

from eczis.schemas import VerifyReport
from eczis.seal.canonical import canonical_json, sha256_bytes
from eczis.seal.manifest import build_manifest


def seal_directory(root: Path, key: bytes, output: Path) -> Path:
    manifest = build_manifest(root)
    manifest_bytes = canonical_json(manifest)
    signature = hmac.new(key, manifest_bytes, hashlib.sha256).hexdigest()
    (root / "manifest.json").write_bytes(manifest_bytes)
    (root / "seal.hmac").write_text(signature + "\n", encoding="ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(p for p in root.rglob("*") if p.is_file() and p != output):
                archive.write(path, path.relative_to(root).as_posix())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def verify_bundle(path: Path, key: bytes) -> VerifyReport:
    errors: list[str] = []
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = set(archive.namelist())
            if "manifest.json" not in names or "seal.hmac" not in names:
                return VerifyReport(False, ("manifest or seal missing",), None)
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(manifest_bytes)
            expected_signature = archive.read("seal.hmac").decode("ascii").strip()
            actual_signature = hmac.new(key, canonical_json(manifest), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected_signature, actual_signature):
                errors.append("HMAC verification failed")
            listed = set()
            digests: list[str] = []
            for item in manifest.get("files", []):
                name = str(item["path"])
                listed.add(name)
                if name not in names:
                    errors.append(f"missing artifact: {name}")
                    continue
                data = archive.read(name)
                digest = sha256_bytes(data)
                digests.append(digest)
                if digest != item["sha256"]:
                    errors.append(f"hash mismatch: {name}")
                if len(data) != item["size"]:
                    errors.append(f"size mismatch: {name}")
            extras = names - listed - {"manifest.json", "seal.hmac"}
            if extras:
                errors.append("unmanifested artifacts: " + ", ".join(sorted(extras)))
            aggregate = hashlib.sha256("".join(digests).encode("ascii")).hexdigest()
            if aggregate != manifest.get("aggregate_sha256"):
                errors.append("aggregate digest mismatch")
            return VerifyReport(not errors, tuple(errors), sha256_bytes(canonical_json(manifest)))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, KeyError) as exc:
        return VerifyReport(False, (f"bundle parse failure: {exc}",), None)
