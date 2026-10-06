# UNFILED-IP
"""Attack matrix for the evidence seal and ledger.

Pre-registered expectations: every single-artifact mutation, extra file, and wrong key
is rejected; a sealed ledger cannot be truncated without detection; a bare (unsealed)
ledger CAN be tail-truncated undetected; and the DEMO development key is derivable from
the published root seed, so a demo bundle can be re-sealed by anyone (by design).
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from eczis.seal.bundle import seal_directory, verify_bundle
from eczis.seal.keys import create_key, development_key
from eczis.seal.ledger import append_event, verify_ledger


def _rewrite(bundle: Path, out: Path, edit: dict[str, bytes | None]) -> Path:
    with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name in edit:
                if edit[name] is not None:
                    dst.writestr(name, edit[name])  # type: ignore[arg-type]
            else:
                dst.writestr(name, src.read(name))
        for name, data in edit.items():
            if name not in src.namelist() and data is not None:
                dst.writestr(name, data)
    return out


@pytest.fixture(scope="module")
def sealed(demo_run: tuple[Path, dict[str, object]]) -> tuple[Path, bytes]:
    output, _ = demo_run
    return output / "eczis_demo.eczis", (output / "DEMO_ONLY.key").read_bytes()


@pytest.mark.parametrize(
    "member",
    ["provenance.npy", "confidence.npy", "map_reconstruction.npy", "provenance.png", "capture.json", "ledger.jsonl"],
)
def test_single_artifact_mutation_is_rejected(sealed: tuple[Path, bytes], tmp_path: Path, member: str) -> None:
    bundle, key = sealed
    with zipfile.ZipFile(bundle) as archive:
        data = bytearray(archive.read(member))
    data[len(data) // 2] ^= 0x01
    report = verify_bundle(_rewrite(bundle, tmp_path / "m.eczis", {member: bytes(data)}), key)
    assert not report.valid
    assert any(member in e for e in report.errors)


def test_deleted_and_extra_artifacts_are_rejected(sealed: tuple[Path, bytes], tmp_path: Path) -> None:
    bundle, key = sealed
    assert not verify_bundle(_rewrite(bundle, tmp_path / "d.eczis", {"truth.npy": None}), key).valid
    assert not verify_bundle(_rewrite(bundle, tmp_path / "x.eczis", {"extra.txt": b"x"}), key).valid


def test_wrong_key_and_garbage_are_rejected(sealed: tuple[Path, bytes], tmp_path: Path) -> None:
    bundle, key = sealed
    assert not verify_bundle(bundle, bytes(32)).valid
    junk = tmp_path / "junk.eczis"
    junk.write_bytes(b"not a zip")
    assert not verify_bundle(junk, key).valid


def test_sealed_ledger_truncation_is_detected(sealed: tuple[Path, bytes], tmp_path: Path) -> None:
    bundle, key = sealed
    with zipfile.ZipFile(bundle) as archive:
        lines = archive.read("ledger.jsonl").decode().splitlines()
    truncated = ("\n".join(lines[:-1]) + "\n").encode()
    assert not verify_bundle(_rewrite(bundle, tmp_path / "t.eczis", {"ledger.jsonl": truncated}), key).valid


def test_bare_ledger_edits_detected_but_tail_truncation_is_not(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    for i in range(4):
        append_event(ledger, "e", {"i": i})
    lines = ledger.read_text().splitlines()
    edited = json.loads(lines[1])
    edited["payload"]["i"] = 99
    for bad in (
        [lines[0], json.dumps(edited), *lines[2:]],
        [lines[0], *lines[2:]],
        [lines[1], lines[0], *lines[2:]],
    ):
        ledger.write_text("\n".join(bad) + "\n")
        assert not verify_ledger(ledger)[0]
    # KNOWN LIMITATION: an unanchored hash chain cannot detect removal of its tail.
    ledger.write_text("\n".join(lines[:2]) + "\n")
    assert verify_ledger(ledger)[0]


def test_development_key_allows_reseal_by_anyone_random_key_does_not(
    demo_run: tuple[Path, dict[str, object]], tmp_path: Path
) -> None:
    output, summary = demo_run
    work = tmp_path / "artifacts"
    shutil.copytree(output / "artifacts", work)
    (work / "provenance.png").write_bytes(b"forged")
    (work / "manifest.json").unlink()
    (work / "seal.hmac").unlink()
    # The root seed is published in RUN_SUMMARY.json and the ledger; the demo key derives from it.
    forged = seal_directory(work, development_key(int(summary["root_seed"])), tmp_path / "f.eczis")  # type: ignore[call-overload]
    assert verify_bundle(forged, (output / "DEMO_ONLY.key").read_bytes()).valid
    attacker = create_key(tmp_path / "attacker.key")
    reforged = seal_directory(work, attacker, tmp_path / "g.eczis")
    assert not verify_bundle(reforged, (output / "DEMO_ONLY.key").read_bytes()).valid
