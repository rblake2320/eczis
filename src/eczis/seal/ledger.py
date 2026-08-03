# UNFILED-IP
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from eczis.seal.canonical import canonical_json, sha256_bytes


def append_event(path: Path, event_type: str, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = "0" * 64
    sequence = 0
    if path.exists() and path.stat().st_size:
        last = path.read_text(encoding="utf-8").splitlines()[-1]
        decoded = json.loads(last)
        previous = str(decoded["entry_hash"])
        sequence = int(decoded["sequence"]) + 1
    body = {"sequence": sequence, "previous_hash": previous, "event_type": event_type, "payload": payload}
    entry_hash = sha256_bytes(canonical_json(body))
    record = {**body, "entry_hash": entry_hash}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return entry_hash


def verify_ledger(path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    previous = "0" * 64
    expected_sequence = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        record = json.loads(line)
        body = {k: record[k] for k in ("sequence", "previous_hash", "event_type", "payload")}
        calculated = sha256_bytes(canonical_json(body))
        if record["sequence"] != expected_sequence:
            errors.append(f"line {line_number}: sequence mismatch")
        if record["previous_hash"] != previous:
            errors.append(f"line {line_number}: previous hash mismatch")
        if record["entry_hash"] != calculated:
            errors.append(f"line {line_number}: entry hash mismatch")
        previous = str(record["entry_hash"])
        expected_sequence += 1
    return not errors, errors
