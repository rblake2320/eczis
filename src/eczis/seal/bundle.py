# UNFILED-IP
from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from eczis.atomic import atomic_write_bytes
from eczis.schemas import VerifyReport
from eczis.seal.canonical import canonical_json, sha256_bytes
from eczis.seal.manifest import RESERVED_NAMES, aggregate_digest, build_manifest
from eczis.versioning import SchemaVersionError, check_schema_version

MAX_MEMBER_BYTES = 512 * 1024 * 1024


def _seal_signature(key: bytes, manifest_bytes: bytes) -> str:
    if len(key) < 16:
        raise ValueError("seal key is too short")
    return hmac.new(key, manifest_bytes, hashlib.sha256).hexdigest()


def seal_directory(root: Path, key: bytes, output: Path) -> Path:
    """Seal ``root`` into ``output`` atomically.

    The archive is built in a sibling temporary file and moved into place only
    after it is complete and fsynced, so an interrupted seal leaves no file
    under ``output`` (and never a partial bundle that could verify).
    """
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("bundle output must not be inside the sealed directory")
    manifest = build_manifest(root)
    manifest_bytes = canonical_json(manifest)
    signature = _seal_signature(key, manifest_bytes)
    atomic_write_bytes(root / "manifest.json", manifest_bytes)
    atomic_write_bytes(root / "seal.hmac", (signature + "\n").encode("ascii"))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    members = [item["path"] for item in manifest["files"]] + sorted(RESERVED_NAMES)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in members:
                source = root / name
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, source.read_bytes(), compresslevel=6)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _unsafe_member(name: str) -> bool:
    posix = PurePosixPath(name)
    return (
        name.startswith(("/", "\\"))
        or "\\" in name
        or ".." in posix.parts
        or (len(name) > 1 and name[1] == ":")
    )


def verify_bundle(path: Path, key: bytes) -> VerifyReport:
    errors: list[str] = []
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            name_list = [info.filename for info in infos]
            names = set(name_list)
            if len(names) != len(name_list):
                return VerifyReport(False, ("duplicate archive members",), None)
            unsafe = sorted(n for n in names if _unsafe_member(n))
            if unsafe:
                return VerifyReport(False, ("unsafe archive member paths: " + ", ".join(unsafe),), None)
            if any(info.file_size > MAX_MEMBER_BYTES for info in infos):
                return VerifyReport(False, ("archive member exceeds size limit",), None)
            if "manifest.json" not in names or "seal.hmac" not in names:
                return VerifyReport(False, ("manifest or seal missing",), None)
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, dict):
                return VerifyReport(False, ("manifest is not an object",), None)
            expected_signature = archive.read("seal.hmac").decode("ascii").strip()
            actual_signature = _seal_signature(key, canonical_json(manifest))
            if not hmac.compare_digest(expected_signature, actual_signature):
                errors.append("HMAC verification failed")
            try:
                check_schema_version(manifest.get("schema_version"), kind="manifest")
            except SchemaVersionError as exc:
                errors.append(str(exc))
            if manifest.get("algorithm") != "HMAC-SHA-256":
                errors.append("unsupported seal algorithm")
            files = manifest.get("files", [])
            paths = [str(item["path"]) for item in files]
            if paths != sorted(paths) or len(set(paths)) != len(paths):
                errors.append("manifest file list is not canonical (sorted, unique)")
            listed: set[str] = set()
            digests: list[str] = []
            for item in files:
                name = str(item["path"])
                listed.add(name)
                if name in RESERVED_NAMES:
                    errors.append(f"reserved name listed as artifact: {name}")
                    continue
                if name not in names:
                    errors.append(f"missing artifact: {name}")
                    continue
                data = archive.read(name)
                digest = sha256_bytes(data)
                digests.append(str(item["sha256"]))
                if digest != item["sha256"]:
                    errors.append(f"hash mismatch: {name}")
                if len(data) != item["size"]:
                    errors.append(f"size mismatch: {name}")
            extras = names - listed - RESERVED_NAMES
            if extras:
                errors.append("unmanifested artifacts: " + ", ".join(sorted(extras)))
            if aggregate_digest(digests) != manifest.get("aggregate_sha256"):
                errors.append("aggregate digest mismatch")
            return VerifyReport(not errors, tuple(errors), sha256_bytes(canonical_json(manifest)))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        return VerifyReport(False, (f"bundle parse failure: {exc.__class__.__name__}: {exc}",), None)
