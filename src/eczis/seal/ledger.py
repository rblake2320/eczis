# UNFILED-IP
"""Append-only, hash-chained JSONL event ledger.

Guarantees:
- appends are serialized across processes by an exclusive lock file;
- each record is one ``\\n``-terminated line, flushed and fsynced;
- a torn final line (crash mid-append) is detected; ``repair_ledger`` moves
  it to a quarantine file and never drops a complete record;
- ``write_anchor`` HMAC-seals the head (sequence + entry hash); verifying
  against an anchor detects tail truncation, which a bare chain cannot.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eczis.atomic import atomic_write_bytes, file_lock
from eczis.seal.canonical import canonical_json, sha256_bytes

GENESIS = "0" * 64
_BODY_KEYS = ("sequence", "previous_hash", "event_type", "payload")
ANCHOR_SCHEMA = "eczis.ledger-anchor/1.0"


class LedgerCorruptError(RuntimeError):
    """Raised when a ledger cannot be appended to without hiding corruption."""


@dataclass(frozen=True, slots=True)
class LedgerHead:
    sequence: int
    entry_hash: str
    count: int


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _read_complete_lines(path: Path) -> tuple[list[str], bytes]:
    """Return complete lines and any trailing bytes not terminated by ``\\n``."""
    if not path.exists():
        return [], b""
    raw = path.read_bytes()
    cut = raw.rfind(b"\n") + 1
    complete, torn = raw[:cut], raw[cut:]
    return complete.decode("utf-8").splitlines(), torn


def _head_from_lines(lines: list[str]) -> LedgerHead:
    if not lines:
        return LedgerHead(-1, GENESIS, 0)
    decoded = json.loads(lines[-1])
    return LedgerHead(int(decoded["sequence"]), str(decoded["entry_hash"]), len(lines))


def append_event(path: Path, event_type: str, payload: dict[str, Any]) -> str:
    if not event_type or not isinstance(event_type, str):
        raise ValueError("event_type must be a non-empty string")
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(_lock_path(path)):
        lines, torn = _read_complete_lines(path)
        if torn:
            raise LedgerCorruptError(
                f"{path}: torn final record ({len(torn)} bytes); run repair_ledger before appending"
            )
        try:
            head = _head_from_lines(lines)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise LedgerCorruptError(f"{path}: unreadable head record: {exc}") from exc
        body = {
            "sequence": head.sequence + 1,
            "previous_hash": head.entry_hash,
            "event_type": event_type,
            "payload": payload,
        }
        entry_hash = sha256_bytes(canonical_json(body))
        record = {**body, "entry_hash": entry_hash}
        line = (json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
        with path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    return entry_hash


def ledger_head(path: Path) -> LedgerHead:
    lines, _ = _read_complete_lines(path)
    return _head_from_lines(lines)


def repair_ledger(path: Path) -> int:
    """Quarantine a torn final record. Returns the number of bytes quarantined."""
    with file_lock(_lock_path(path)):
        raw = path.read_bytes() if path.exists() else b""
        cut = raw.rfind(b"\n") + 1
        torn = raw[cut:]
        if not torn:
            return 0
        quarantine = path.with_name(path.name + ".torn")
        with quarantine.open("ab") as handle:
            handle.write(torn + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_write_bytes(path, raw[:cut])
        return len(torn)


def _anchor_mac(key: bytes, head: dict[str, Any]) -> str:
    return hmac.new(key, canonical_json(head), hashlib.sha256).hexdigest()


def write_anchor(path: Path, key: bytes, anchor_path: Path | None = None) -> Path:
    """HMAC-seal the current ledger head so truncation becomes detectable."""
    with file_lock(_lock_path(path)):
        head = ledger_head(path)
    body = {"schema": ANCHOR_SCHEMA, "sequence": head.sequence, "entry_hash": head.entry_hash, "count": head.count}
    target = anchor_path or path.with_name(path.name + ".anchor.json")
    atomic_write_bytes(target, canonical_json({**body, "hmac_sha256": _anchor_mac(key, body)}))
    return target


def verify_ledger(
    path: Path,
    *,
    anchor_path: Path | None = None,
    key: bytes | None = None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not path.exists():
        return False, ["ledger missing"]
    lines, torn = _read_complete_lines(path)
    if torn:
        errors.append(f"torn final record ({len(torn)} bytes)")
    previous = GENESIS
    expected_sequence = 0
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
            body = {k: record[k] for k in _BODY_KEYS}
            stored_hash = str(record["entry_hash"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"line {line_number}: malformed record ({exc.__class__.__name__})")
            expected_sequence += 1
            previous = "<unknown>"
            continue
        if set(record) != set(_BODY_KEYS) | {"entry_hash"}:
            errors.append(f"line {line_number}: unexpected fields")
        calculated = sha256_bytes(canonical_json(body))
        if record["sequence"] != expected_sequence:
            errors.append(f"line {line_number}: sequence mismatch")
        if record["previous_hash"] != previous:
            errors.append(f"line {line_number}: previous hash mismatch")
        if stored_hash != calculated:
            errors.append(f"line {line_number}: entry hash mismatch")
        previous = stored_hash
        expected_sequence += 1
    if anchor_path is not None:
        if key is None:
            errors.append("anchor supplied without key")
        elif not anchor_path.exists():
            errors.append("anchor missing")
        else:
            try:
                anchor = json.loads(anchor_path.read_bytes())
                mac = str(anchor.pop("hmac_sha256"))
            except (json.JSONDecodeError, KeyError, TypeError):
                errors.append("anchor malformed")
            else:
                if not hmac.compare_digest(mac, _anchor_mac(key, anchor)):
                    errors.append("anchor HMAC verification failed")
                elif anchor.get("schema") != ANCHOR_SCHEMA:
                    errors.append("anchor schema unsupported")
                else:
                    if int(anchor["count"]) > len(lines):
                        errors.append(
                            f"ledger truncated: anchor records {anchor['count']} entries, found {len(lines)}"
                        )
                    elif int(anchor["count"]) >= 1:
                        index = int(anchor["count"]) - 1
                        try:
                            anchored = json.loads(lines[index])["entry_hash"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            anchored = None
                        if anchored != anchor["entry_hash"]:
                            errors.append("anchored entry hash does not match ledger")
    return not errors, errors
