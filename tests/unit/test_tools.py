# UNFILED-IP
"""T0: disclosure gate, evidence guard, executed claims gate, release verifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import claims_check  # noqa: E402
import evidence_guard  # noqa: E402
import ip_gate  # noqa: E402
import verify_release  # noqa: E402


def _junit(path: Path, cases: dict[str, str]) -> Path:
    body = []
    for name, outcome in cases.items():
        inner = {"passed": "", "failed": "<failure message='x'/>", "skipped": "<skipped message='nope'/>",
                 "na": "<skipped message='platform-not-applicable: windows only'/>"}[outcome]
        body.append(f"<testcase classname='t' name='{name}'>{inner}</testcase>")
    path.write_text(f"<testsuites><testsuite>{''.join(body)}</testsuite></testsuites>", encoding="utf-8")
    return path


def _repo(tmp_path: Path, registry: str, tests: str) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "RELEASE_CLAIMS.yaml").write_text(registry, encoding="utf-8")
    (tmp_path / "tests" / "test_x.py").write_text(tests, encoding="utf-8")
    return tmp_path


REG = "claims:\n  - id: C1\n    text: Bounded claim.\n    tests: [test_a, test_b]\n"
TESTS = "def test_a():\n    pass\n\ndef test_b():\n    pass\n"


def test_claims_static_pass_and_missing_test(tmp_path: Path) -> None:
    repo = _repo(tmp_path, REG, TESTS)
    assert claims_check.main(["--root", str(repo)]) == 0
    (repo / "tests" / "test_x.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    assert claims_check.main(["--root", str(repo)]) == 2


def test_claims_reject_forbidden_language_duplicates_and_ambiguity(tmp_path: Path) -> None:
    reg = REG + "  - id: C1\n    text: Delivers infinite resolution.\n    tests: [test_a]\n"
    repo = _repo(tmp_path, reg, TESTS + "\ndef test_a():\n    pass\n")
    errors = claims_check.static_errors(claims_check.parse_registry(reg), (repo / "tests" / "test_x.py").read_text())
    assert any("duplicate claim id" in e for e in errors)
    assert any("forbidden" in e for e in errors)
    assert any("ambiguous" in e for e in errors)


@pytest.mark.parametrize(("outcome", "code"), [("passed", 0), ("failed", 2), ("skipped", 2)])
def test_claims_executed_mode(tmp_path: Path, outcome: str, code: int) -> None:
    repo = _repo(tmp_path, REG, TESTS)
    junit = _junit(tmp_path / "r.xml", {"test_a": "passed", "test_b": outcome})
    evidence = tmp_path / "evidence.json"
    assert claims_check.main(["--root", str(repo), "--junit", str(junit), "--evidence", str(evidence)]) == code
    record = json.loads(evidence.read_text(encoding="utf-8"))
    assert record["passed"] is (code == 0)
    assert len(record["evidence_sha256"]) == 64


def test_claims_executed_mode_flags_unexecuted_test(tmp_path: Path) -> None:
    repo = _repo(tmp_path, REG, TESTS)
    junit = _junit(tmp_path / "r.xml", {"test_a": "passed"})
    assert claims_check.main(["--root", str(repo), "--junit", str(junit)]) == 2


@pytest.mark.parametrize(
    "path",
    ["evidence/family/kid.jpg", "photo.HEIC", "artifacts/demo/x.png", "a/b.eczis", "seal.key", ".env",
     "private/notes.txt", "data/frame.npy", "raw/IMG_0001.CR3"],
)
def test_evidence_guard_blocks_personal_media_and_secrets(path: str) -> None:
    assert evidence_guard.violations([path])


@pytest.mark.parametrize("path", ["src/eczis/cli.py", "docs/ARCHITECTURE.md", ".env.example", "viewer/index.html"])
def test_evidence_guard_allows_source(path: str) -> None:
    assert not evidence_guard.violations([path])


def test_repository_passes_evidence_guard() -> None:
    assert evidence_guard.main([]) == 0


def test_ip_gate_detects_missing_header_and_open_status(tmp_path: Path) -> None:
    (tmp_path / "IP_STATUS.yaml").write_text("# UNFILED-IP\nstatus: UNFILED\npublic_disclosure_allowed: false\n",
                                             encoding="utf-8")
    (tmp_path / "ok.py").write_text("# UNFILED-IP\n", encoding="utf-8")
    assert ip_gate.main(tmp_path) == 0
    (tmp_path / "leak.md").write_text("# Public notes\n", encoding="utf-8")
    assert ip_gate.main(tmp_path) == 2
    (tmp_path / "leak.md").unlink()
    (tmp_path / "IP_STATUS.yaml").write_text("status: FILED\npublic_disclosure_allowed: true\n", encoding="utf-8")
    assert ip_gate.main(tmp_path) == 2


def test_release_verifier_classifies_skips(tmp_path: Path) -> None:
    junit = _junit(tmp_path / "r.xml", {"a": "passed", "b": "na", "c": "skipped", "d": "failed"})
    totals, problems = verify_release.junit_totals(junit)
    assert totals == {"tests": 4, "passed": 1, "failed": 1, "errors": 0, "skipped": 1, "skipped_platform_na": 1}
    assert any("unclassified skip" in p for p in problems)
    assert any(p.startswith("failed") for p in problems)
    empty = tmp_path / "e.xml"
    empty.write_text("<testsuites/>", encoding="utf-8")
    assert verify_release.junit_totals(empty)[1] == ["JUnit report contains no test cases"]


def test_pre_push_hook_blocks_foreign_remote() -> None:
    hook = (ROOT / "tools" / "hooks" / "pre-push").read_text(encoding="utf-8")
    assert "rblake2320/eczis" in hook and "exit 1" in hook
    for gate in ("evidence_guard.py", "ip_gate.py", "claims_check.py"):
        assert gate in hook
