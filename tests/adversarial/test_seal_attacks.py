# UNFILED-IP
"""Bundle and ledger tamper, crash, and replay attacks (T8)."""

from __future__ import annotations

import json
import multiprocessing
import zipfile
from pathlib import Path

import pytest

from eczis.seal.bundle import seal_directory, verify_bundle
from eczis.seal.canonical import canonical_json
from eczis.seal.keys import development_key
from eczis.seal.ledger import (
    LedgerCorruptError,
    append_event,
    ledger_head,
    repair_ledger,
    verify_ledger,
    write_anchor,
)

KEY = development_key(42)
FIXED_ENV = {"fixed": True}


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"alpha evidence")
    (root / "sub" / "b.bin").write_bytes(bytes(range(256)))
    return root


def _rewrite(src: Path, dst: Path, *, mutate: dict[str, bytes | None] | None = None,
             extra: dict[str, bytes] | None = None) -> Path:
    mutate = mutate or {}
    with zipfile.ZipFile(src) as source, zipfile.ZipFile(dst, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename in mutate:
                replacement = mutate[item.filename]
                if replacement is None:
                    continue
                data = replacement
            target.writestr(item.filename, data)
        for name, data in (extra or {}).items():
            # Set the raw name after construction: on Windows ZipInfo() rewrites os.sep
            # ("\\") to "/", which would silently defang a backslash-path attack.
            info = zipfile.ZipInfo("placeholder")
            info.filename = name
            target.writestr(info, data)
    return dst


@pytest.fixture()
def sealed(tmp_path: Path) -> Path:
    return seal_directory(_root(tmp_path), KEY, tmp_path / "clean.eczis")


def test_clean_bundle_verifies(sealed: Path) -> None:
    report = verify_bundle(sealed, KEY)
    assert report.valid, report.errors
    assert report.manifest_digest


def test_single_bit_flip_detected(sealed: Path, tmp_path: Path) -> None:
    with zipfile.ZipFile(sealed) as z:
        data = bytearray(z.read("sub/b.bin"))
    data[100] ^= 0x01
    report = verify_bundle(_rewrite(sealed, tmp_path / "flip.eczis", mutate={"sub/b.bin": bytes(data)}), KEY)
    assert not report.valid
    assert any("hash mismatch" in e for e in report.errors)


def test_deletion_detected(sealed: Path, tmp_path: Path) -> None:
    report = verify_bundle(_rewrite(sealed, tmp_path / "del.eczis", mutate={"a.txt": None}), KEY)
    assert not report.valid
    assert any("missing artifact" in e for e in report.errors)


def test_insertion_detected(sealed: Path, tmp_path: Path) -> None:
    report = verify_bundle(_rewrite(sealed, tmp_path / "ins.eczis", extra={"planted.txt": b"x"}), KEY)
    assert not report.valid
    assert any("unmanifested" in e for e in report.errors)


def test_manifest_reorder_or_edit_breaks_hmac(sealed: Path, tmp_path: Path) -> None:
    with zipfile.ZipFile(sealed) as z:
        manifest = json.loads(z.read("manifest.json"))
    manifest["files"].reverse()
    report = verify_bundle(
        _rewrite(sealed, tmp_path / "reorder.eczis", mutate={"manifest.json": json.dumps(manifest).encode()}), KEY
    )
    assert not report.valid
    assert "HMAC verification failed" in report.errors


def test_attacker_resigned_noncanonical_manifest_rejected(sealed: Path, tmp_path: Path) -> None:
    """Even with the key, a manifest that is unsorted/duplicated is not accepted."""
    import hashlib
    import hmac

    with zipfile.ZipFile(sealed) as z:
        manifest = json.loads(z.read("manifest.json"))
    manifest["files"] = manifest["files"] + manifest["files"][:1]
    sig = hmac.new(KEY, canonical_json(manifest), hashlib.sha256).hexdigest()
    forged = _rewrite(
        sealed, tmp_path / "dup.eczis",
        mutate={"manifest.json": canonical_json(manifest), "seal.hmac": (sig + "\n").encode()},
    )
    report = verify_bundle(forged, KEY)
    assert not report.valid
    assert any("canonical" in e for e in report.errors)


def test_unknown_schema_major_fails_closed(sealed: Path, tmp_path: Path) -> None:
    import hashlib
    import hmac

    with zipfile.ZipFile(sealed) as z:
        manifest = json.loads(z.read("manifest.json"))
    manifest["schema_version"] = "2.0"
    sig = hmac.new(KEY, canonical_json(manifest), hashlib.sha256).hexdigest()
    forged = _rewrite(
        sealed, tmp_path / "v2.eczis",
        mutate={"manifest.json": canonical_json(manifest), "seal.hmac": (sig + "\n").encode()},
    )
    report = verify_bundle(forged, KEY)
    assert not report.valid
    assert any("unsupported schema major" in e for e in report.errors)


def test_manifest_order_is_codepoint_canonical_with_mixed_case(tmp_path: Path) -> None:
    """Regression (Windows CI): Path sorting is case-insensitive there; manifests must not be."""
    root = tmp_path / "mixed"
    root.mkdir()
    for name in ("b.json", "REPORT.md", "a.txt", "Z.png"):
        (root / name).write_bytes(name.encode())
    bundle = seal_directory(root, KEY, tmp_path / "mixed.eczis")
    with zipfile.ZipFile(bundle) as z:
        paths = [item["path"] for item in json.loads(z.read("manifest.json"))["files"]]
    assert paths == ["REPORT.md", "Z.png", "a.txt", "b.json"]
    assert verify_bundle(bundle, KEY).valid


def test_wrong_key_fails_cleanly(sealed: Path) -> None:
    report = verify_bundle(sealed, development_key(43))
    assert not report.valid
    assert "HMAC verification failed" in report.errors


@pytest.mark.parametrize("name", ["../escape.txt", "/abs.txt", "C:/win.txt", "back\\slash.txt"])
def test_unsafe_member_paths_rejected(sealed: Path, tmp_path: Path, name: str) -> None:
    report = verify_bundle(_rewrite(sealed, tmp_path / "unsafe.eczis", extra={name: b"x"}), KEY)
    assert not report.valid
    assert any("unsafe" in e for e in report.errors)


def test_duplicate_members_rejected(sealed: Path, tmp_path: Path) -> None:
    dup = tmp_path / "dupmember.eczis"
    with pytest.warns(UserWarning):
        _rewrite(sealed, dup, extra={"a.txt": b"shadow"})
    report = verify_bundle(dup, KEY)
    assert not report.valid
    assert "duplicate archive members" in report.errors


def test_truncated_or_garbage_bundle_fails_closed(sealed: Path, tmp_path: Path) -> None:
    truncated = tmp_path / "trunc.eczis"
    truncated.write_bytes(sealed.read_bytes()[:-40])
    assert not verify_bundle(truncated, KEY).valid
    garbage = tmp_path / "garbage.eczis"
    garbage.write_bytes(b"not a zip")
    assert not verify_bundle(garbage, KEY).valid
    assert not verify_bundle(tmp_path / "absent.eczis", KEY).valid


def test_interrupted_seal_leaves_no_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    output = tmp_path / "out" / "bundle.eczis"
    calls = {"n": 0}
    original = zipfile.ZipFile.writestr

    def crash(self: zipfile.ZipFile, *args: object, **kwargs: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated power loss")
        original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(zipfile.ZipFile, "writestr", crash)
    with pytest.raises(OSError, match="power loss"):
        seal_directory(root, KEY, output)
    assert not output.exists()
    assert list(output.parent.iterdir()) == []  # no orphaned temporary file


def test_interrupted_replace_keeps_previous_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    output = tmp_path / "bundle.eczis"
    seal_directory(root, KEY, output)
    before = output.read_bytes()
    (root / "a.txt").write_bytes(b"changed")

    import os

    real_replace = os.replace
    hits = {"bundle_rename": 0}

    def fail_on_bundle(src: object, dst: object) -> None:
        if Path(str(dst)) == output:
            hits["bundle_rename"] += 1
            raise OSError("crash during final bundle rename")
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr("eczis.seal.bundle.os.replace", fail_on_bundle)
    with pytest.raises(OSError, match="final bundle rename"):
        seal_directory(root, KEY, output)
    assert hits["bundle_rename"] == 1  # crash point is the final rename, after the archive was complete
    assert output.read_bytes() == before
    assert verify_bundle(output, KEY).valid


def test_same_platform_replay_is_bit_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("eczis.seal.manifest.replay_environment", lambda: FIXED_ENV)
    root = _root(tmp_path)
    first = seal_directory(root, KEY, tmp_path / "one.eczis").read_bytes()
    second = seal_directory(root, KEY, tmp_path / "two.eczis").read_bytes()
    assert first == second


def test_output_inside_sealed_directory_refused(tmp_path: Path) -> None:
    root = _root(tmp_path)
    with pytest.raises(ValueError, match="inside"):
        seal_directory(root, KEY, root / "self.eczis")


def test_short_key_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="short"):
        seal_directory(_root(tmp_path), b"tiny", tmp_path / "x.eczis")


# ---------------------------------------------------------------- ledger ----


def _ledger(tmp_path: Path, n: int = 4) -> Path:
    path = tmp_path / "ledger.jsonl"
    for i in range(n):
        append_event(path, f"event{i}", {"i": i})
    return path


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


def test_ledger_middle_deletion_and_reorder_detected(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2], lines[3]]) + "\n", encoding="utf-8")
    assert not verify_ledger(path)[0]
    path.write_text("\n".join([lines[1], lines[0], lines[2], lines[3]]) + "\n", encoding="utf-8")
    assert not verify_ledger(path)[0]


def test_ledger_tail_truncation_needs_anchor(tmp_path: Path) -> None:
    """A bare chain cannot see tail truncation; the HMAC anchor can."""
    path = _ledger(tmp_path)
    anchor = write_anchor(path, KEY)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    assert verify_ledger(path)[0], "documented limitation: chain alone accepts a truncated prefix"
    ok, errors = verify_ledger(path, anchor_path=anchor, key=KEY)
    assert not ok
    assert any("truncated" in e for e in errors)


def test_ledger_forged_anchor_rejected(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    anchor = write_anchor(path, KEY)
    data = json.loads(anchor.read_text(encoding="utf-8"))
    data["count"] = 2
    anchor.write_text(json.dumps(data), encoding="utf-8")
    ok, errors = verify_ledger(path, anchor_path=anchor, key=KEY)
    assert not ok
    assert "anchor HMAC verification failed" in errors
    assert not verify_ledger(path, anchor_path=write_anchor(path, development_key(1)), key=KEY)[0]


def test_ledger_rewritten_history_with_valid_chain_caught_by_anchor(tmp_path: Path) -> None:
    """Attacker rebuilds a fully self-consistent chain; only the anchored head reveals it."""
    path = _ledger(tmp_path)
    anchor = write_anchor(path, KEY)
    path.unlink()
    for i in range(4):
        append_event(path, f"forged{i}", {"i": i})
    assert verify_ledger(path)[0]
    ok, errors = verify_ledger(path, anchor_path=anchor, key=KEY)
    assert not ok
    assert "anchored entry hash does not match ledger" in errors


def test_torn_tail_detected_blocks_append_and_repairs(tmp_path: Path) -> None:
    path = _ledger(tmp_path, 3)
    with path.open("ab") as handle:
        handle.write(b'{"sequence":3,"previous_ha')  # crash mid-append
    ok, errors = verify_ledger(path)
    assert not ok and any("torn" in e for e in errors)
    with pytest.raises(LedgerCorruptError, match="torn"):
        append_event(path, "after-crash", {})
    assert repair_ledger(path) > 0
    assert (tmp_path / "ledger.jsonl.torn").is_file()
    assert verify_ledger(path)[0]
    assert ledger_head(path).count == 3  # no complete record was dropped
    append_event(path, "after-repair", {})
    assert ledger_head(path).count == 4


def test_ledger_malformed_and_injected_fields_detected(tmp_path: Path) -> None:
    path = _ledger(tmp_path, 2)
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["note"] = "injected"
    path.write_text(lines[0] + "\n" + json.dumps(record) + "\nnot json\n", encoding="utf-8")
    ok, errors = verify_ledger(path)
    assert not ok
    assert any("unexpected fields" in e for e in errors)
    assert any("malformed" in e for e in errors)


def _append_many(args: tuple[str, int]) -> None:
    path, worker = args
    for i in range(15):
        append_event(Path(path), "concurrent", {"worker": worker, "i": i})


def test_concurrent_appends_from_processes_keep_chain_valid(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(4) as pool:
        pool.map(_append_many, [(str(path), w) for w in range(4)])
    ok, errors = verify_ledger(path)
    assert ok, errors
    assert ledger_head(path).count == 60
