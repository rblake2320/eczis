# UNFILED-IP
"""Release verifier: one command that decides whether a build may be called a release.

Runs the IP gate, evidence guard, and executed claims gate; checks the JUnit
report for failures and for skips that are not explicitly classified
``platform-not-applicable``; optionally verifies a sealed demo bundle; and
writes ``release_manifest.json`` with the frozen test count, claim evidence
digest, lockfile digest and commit -- HMAC-sealed when a key is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import claims_check  # noqa: E402
import evidence_guard  # noqa: E402
import ip_gate  # noqa: E402

PLATFORM_NA = "platform-not-applicable"


def junit_totals(path: Path) -> tuple[dict[str, int], list[str]]:
    totals = {"tests": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "skipped_platform_na": 0}
    problems: list[str] = []
    for case in ET.parse(path).getroot().iter("testcase"):
        totals["tests"] += 1
        label = f"{case.get('classname', '')}::{case.get('name', '')}"
        if case.find("failure") is not None:
            totals["failed"] += 1
            problems.append(f"failed: {label}")
        elif case.find("error") is not None:
            totals["errors"] += 1
            problems.append(f"error: {label}")
        elif (skip := case.find("skipped")) is not None:
            message = (skip.get("message") or "") + (skip.text or "")
            if PLATFORM_NA in message:
                totals["skipped_platform_na"] += 1
            else:
                totals["skipped"] += 1
                problems.append(f"unclassified skip (counts as failure): {label}")
        else:
            totals["passed"] += 1
    if totals["tests"] == 0:
        problems.append("JUnit report contains no test cases")
    return totals, problems


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("release_manifest.json"))
    parser.add_argument("--bundle", type=Path, help="sealed .eczis bundle to verify")
    parser.add_argument("--key", type=Path, help="seal key file (raw or DPAPI-wrapped)")
    parser.add_argument("--dev-seed", type=int, help="verify bundle with the deterministic demo key")
    args = parser.parse_args(argv)

    failures: list[str] = []
    if ip_gate.main() != 0:
        failures.append("IP gate failed")
    if evidence_guard.main([]) != 0:
        failures.append("evidence guard failed")
    claims_evidence = args.output.with_name("claims_evidence.json")
    if claims_check.main(["--junit", str(args.junit), "--evidence", str(claims_evidence)]) != 0:
        failures.append("executed claims gate failed")
    totals, problems = junit_totals(args.junit)
    failures.extend(problems)

    sys.path.insert(0, str(ROOT / "src"))
    from eczis import __version__
    from eczis.seal.canonical import canonical_json
    from eczis.seal.keys import development_key, load_key

    key: bytes | None = None
    if args.key is not None:
        key = load_key(args.key)
    elif args.dev_seed is not None:
        key = development_key(args.dev_seed)

    bundle_record: dict[str, Any] | None = None
    if args.bundle is not None:
        from eczis.seal.bundle import verify_bundle

        if key is None:
            failures.append("--bundle requires --key or --dev-seed")
        else:
            report = verify_bundle(args.bundle, key)
            bundle_record = {"path": args.bundle.name, "valid": report.valid, "manifest_digest": report.manifest_digest}
            if not report.valid:
                failures.append("bundle verification failed: " + "; ".join(report.errors))

    lock = ROOT / "uv.lock"
    evidence_digest = None
    if claims_evidence.is_file():
        evidence_digest = json.loads(claims_evidence.read_text(encoding="utf-8")).get("evidence_sha256")
    body: dict[str, Any] = {
        "schema": "eczis.release-manifest/1.0",
        "eczis_version": __version__,
        "commit": _git_commit(),
        "claim_language": "Infinitely sampleable output, finite and explicitly classified evidence.",
        "evidence_level": "synthetic software reduction-to-practice",
        "tests": totals,
        "claims_evidence_sha256": evidence_digest,
        "junit_sha256": hashlib.sha256(args.junit.read_bytes()).hexdigest(),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
        "bundle": bundle_record,
        "release_ready": not failures,
        "failures": failures,
    }
    manifest: dict[str, Any] = dict(body)
    if key is not None:
        manifest["hmac_sha256"] = hmac.new(key, canonical_json(body), hashlib.sha256).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        print("Release verification FAIL:")
        print("\n".join(f"  - {f}" for f in failures))
        return 2
    print(f"Release verification PASS: {totals['passed']}/{totals['tests']} tests passed "
          f"({totals['skipped_platform_na']} platform-not-applicable); manifest -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
