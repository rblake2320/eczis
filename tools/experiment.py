# UNFILED-IP
"""Run the phantom benchmark as experiments: pinned datasets, one change per candidate, losers kept.

    uv run python tools/experiment.py pin-dataset can66 phantom <graded.ply> --trajectory <tum.txt> --diameter-mm 66 --note "..."
    uv run python tools/experiment.py pin-dataset notube negative <graded.ply> --diameter-mm 7
    uv run python tools/experiment.py run can66 baseline
    uv run python tools/experiment.py run can66 optical axis_from=optical
    uv run python tools/experiment.py run can66 roi axial_window=0.2:1.4
    uv run python tools/experiment.py combine can66 optical roi          # compatibility check of two single changes
    uv run python tools/experiment.py report

Every run is appended to artifacts/experiments/log.jsonl with the dataset digest, the declared change, the git
commit and the policy-freeze status. Declare the candidate BEFORE looking at its result; a candidate chosen after
seeing an error is an unblinded run and should say so in its name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eczis.lab.experiments import (  # noqa: E402
    BASELINE_ARGS,
    Candidate,
    baseline_record,
    combined,
    compatibility_verdict,
    load_dataset,
    pin_dataset,
    read_log,
    report,
    run_candidate,
    score,
)


def parse_kv(items: list[str]) -> dict[str, object]:
    args: dict[str, object] = dict(BASELINE_ARGS)
    for it in items:
        k, _, v = it.partition("=")
        if v.lower() in ("true", "false"):
            args[k] = v.lower() == "true"
        else:
            try:
                args[k] = float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v
            except ValueError:
                args[k] = v
    return args


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pin-dataset")
    p.add_argument("name")
    p.add_argument("kind", choices=["phantom", "negative"])
    p.add_argument("ply", type=Path)
    p.add_argument("--trajectory", type=Path, default=None)
    p.add_argument("--diameter-mm", type=float, required=True)
    p.add_argument("--note", default="")
    p.add_argument("--extra", type=Path, nargs="*", default=[])
    p = sub.add_parser("run")
    p.add_argument("dataset")
    p.add_argument("candidate", help="a name; 'baseline' runs the baseline args")
    p.add_argument("knobs", nargs="*", help="key=value, one change from the baseline")
    p = sub.add_parser("combine")
    p.add_argument("dataset")
    p.add_argument("a")
    p.add_argument("b")
    sub.add_parser("report")
    a = ap.parse_args()

    if a.cmd == "pin-dataset":
        ds = pin_dataset(a.name, a.kind, a.ply, a.trajectory, a.diameter_mm, a.note, a.extra)
        print(
            json.dumps({"pinned": ds.name, "digest": ds.digest, "files": len(ds.files)}, indent=2)
        )
        return 0
    if a.cmd == "run":
        ds = load_dataset(a.dataset)
        if a.candidate == "baseline" and a.knobs:
            print("'baseline' takes no knobs; name the candidate after its change", file=sys.stderr)
            return 2
        cand = Candidate(
            a.candidate, dict(BASELINE_ARGS) if a.candidate == "baseline" else parse_kv(a.knobs)
        )
        rec = run_candidate(ds, cand)
        base = baseline_record(ds.name) if rec["change"] != "baseline" else None
        rec["score"] = (
            score(rec, base) if rec["change"] != "baseline" else {"verdict": rec["status"]}
        )
        print(
            json.dumps(
                {
                    k: rec[k]
                    for k in ("run_id", "change", "status", "refusal", "policy", "metrics", "score")
                },
                indent=2,
            )
        )
        return 0
    if a.cmd == "combine":
        ds = load_dataset(a.dataset)
        recs = {r["candidate"]: r for r in read_log() if r["dataset"] == ds.name}
        if a.a not in recs or a.b not in recs:
            print(f"run {a.a!r} and {a.b!r} on {ds.name!r} first", file=sys.stderr)
            return 2
        ca, cb = Candidate(a.a, recs[a.a]["args"]), Candidate(a.b, recs[a.b]["args"])
        rec = run_candidate(ds, combined(ca, cb))
        rec["compatibility"] = compatibility_verdict(recs[a.a], recs[a.b], rec)
        print(
            json.dumps(
                {
                    k: rec[k]
                    for k in ("run_id", "change", "status", "refusal", "metrics", "compatibility")
                },
                indent=2,
            )
        )
        return 0
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
