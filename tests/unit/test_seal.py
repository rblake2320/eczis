# UNFILED-IP
from __future__ import annotations

import zipfile
from pathlib import Path

from eczis.seal.bundle import seal_directory, verify_bundle
from eczis.seal.keys import development_key
from eczis.seal.ledger import append_event, verify_ledger


def test_bundle_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "evidence.txt").write_text("measured evidence", encoding="utf-8")
    key = development_key(42)
    bundle = seal_directory(root, key, tmp_path / "clean.eczis")
    assert verify_bundle(bundle, key).valid
    assert not verify_bundle(bundle, development_key(43)).valid

    tampered = tmp_path / "tampered.eczis"
    with zipfile.ZipFile(bundle, "r") as source, zipfile.ZipFile(tampered, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "evidence.txt":
                data += b"!"
            target.writestr(item, data)
    report = verify_bundle(tampered, key)
    assert not report.valid
    assert any("hash mismatch" in error or "size mismatch" in error for error in report.errors)


def test_ledger_detects_mutation(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    append_event(ledger, "one", {"value": 1})
    append_event(ledger, "two", {"value": 2})
    assert verify_ledger(ledger)[0]
    text = ledger.read_text(encoding="utf-8").replace('"value":2', '"value":9')
    ledger.write_text(text, encoding="utf-8")
    valid, errors = verify_ledger(ledger)
    assert not valid
    assert errors
