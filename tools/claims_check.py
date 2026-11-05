# UNFILED-IP
"""Release-claims gate.

Static mode (default): every claim in RELEASE_CLAIMS.yaml has a unique id,
evidence-bounded text, and >= 1 mapped test that is defined exactly once.

Executed mode (``--junit results.xml``): additionally, every mapped test
must appear in the pytest JUnit report and have PASSED -- missing (stale or
unexecuted), failed, errored, or skipped mapped tests fail the gate.
``--evidence out.json`` writes a digest-bound claim->result record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "infinite resolution",
    "infinite detail",
    "hallucination-free",
    "perfect reconstruction",
    "forensic-grade",
    "ground-truth recovery from one ordinary image",
)


def parse_registry(text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("- id:"):
            current = {"id": line.split(":", 1)[1].strip(), "text": "", "tests": []}
            claims.append(current)
        elif current is not None and line.startswith("text:"):
            current["text"] = line.split(":", 1)[1].strip()
        elif current is not None and line.startswith("tests:"):
            match = re.search(r"\[([^\]]*)\]", line)
            current["tests"] = [t.strip() for t in match.group(1).split(",") if t.strip()] if match else []
    return claims


def static_errors(claims: list[dict[str, Any]], test_text: str) -> list[str]:
    errors: list[str] = []
    if not claims:
        errors.append("RELEASE_CLAIMS.yaml declares no claims")
    ids = [c["id"] for c in claims]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        errors.append(f"duplicate claim id: {dup}")
    for claim in claims:
        if not claim["text"]:
            errors.append(f"{claim['id']}: empty claim text")
        lowered = claim["text"].lower()
        for phrase in FORBIDDEN:
            if phrase in lowered:
                errors.append(f"{claim['id']}: forbidden unqualified phrase {phrase!r}")
        if not claim["tests"]:
            errors.append(f"{claim['id']}: no mapped tests")
        for name in claim["tests"]:
            count = len(re.findall(rf"^\s*def {re.escape(name)}\(", test_text, flags=re.MULTILINE))
            if count == 0:
                errors.append(f"{claim['id']}: mapped test not found: {name}")
            elif count > 1:
                errors.append(f"{claim['id']}: mapped test defined {count} times (ambiguous): {name}")
    return errors


def junit_results(path: Path) -> dict[str, list[str]]:
    """Map bare test function name -> list of outcomes (one per parametrization)."""
    results: dict[str, list[str]] = {}
    root = ET.parse(path).getroot()
    for case in root.iter("testcase"):
        name = case.get("name", "").split("[", 1)[0]
        if case.find("failure") is not None:
            outcome = "failed"
        elif case.find("error") is not None:
            outcome = "error"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        results.setdefault(name, []).append(outcome)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, help="pytest --junitxml report; enables executed mode")
    parser.add_argument("--evidence", type=Path, help="write claim evidence JSON (executed mode)")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    registry = (args.root / "RELEASE_CLAIMS.yaml").read_text(encoding="utf-8")
    claims = parse_registry(registry)
    test_text = "\n".join(p.read_text(encoding="utf-8") for p in sorted((args.root / "tests").rglob("test_*.py")))
    errors = static_errors(claims, test_text)
    required = sorted({t for c in claims for t in c["tests"]})

    evidence: dict[str, Any] | None = None
    if args.junit is not None:
        if not args.junit.is_file():
            errors.append(f"JUnit report not found: {args.junit}")
        else:
            results = junit_results(args.junit)
            for name in required:
                outcomes = results.get(name)
                if not outcomes:
                    errors.append(f"mapped test not executed (stale or deselected): {name}")
                elif any(o != "passed" for o in outcomes):
                    errors.append(f"mapped test did not pass: {name} -> {sorted(set(outcomes))}")
            junit_digest = hashlib.sha256(args.junit.read_bytes()).hexdigest()
            claim_records = [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "tests": {t: results.get(t, ["not-executed"]) for t in c["tests"]},
                }
                for c in claims
            ]
            body = {
                "schema": "eczis.claims-evidence/1.0",
                "junit_sha256": junit_digest,
                "registry_sha256": hashlib.sha256(registry.encode("utf-8")).hexdigest(),
                "claims": claim_records,
                "total_testcases": sum(len(v) for v in results.values()),
                "passed": not errors,
            }
            canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            evidence = {**body, "evidence_sha256": hashlib.sha256(canonical).hexdigest()}

    if evidence is not None and args.evidence is not None:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if errors:
        print("Claims check FAIL:")
        print("\n".join(f"  - {e}" for e in errors))
        return 2
    mode = "executed" if args.junit is not None else "static"
    print(f"Claims check PASS ({mode}): {len(claims)} claims, {len(required)} mapped tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
