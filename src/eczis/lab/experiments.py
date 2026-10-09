# UNFILED-IP
"""Experiment runner for the phantom benchmark: one declared change at a time against a pinned dataset.

Borrowed discipline (owner, 2026-10-08, from the public integer-multiplication exponent race): pin the inputs,
change one thing per candidate, keep every result including the losers, run the negative controls on every
candidate, and test whether two individual winners still win together. Nothing here changes the grading policy
or the benchmark's gates; it only runs them and records what they said.

What is and is not independent: the scorer in this module decides "better / same / worse / control broken"
from the benchmark's recorded numbers. It is a separate decision layer with its own rules, not a second
measurement of the point cloud. An independent re-measurement is a listed gap, not a claim.

EXPERIMENTAL; never changes production M/R/I/U.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools" / "phantom_benchmark.py"
EXPERIMENTS = ROOT / "artifacts" / "experiments"
DATASETS = EXPERIMENTS / "datasets"
RUNS = EXPERIMENTS / "runs"
LOG = EXPERIMENTS / "log.jsonl"  # append-only; losers stay

# the knobs a candidate may declare (benchmark options that change the measurement, never the policy)
KNOBS: dict[str, str] = {
    "axis_from": "--axis-from",
    "axial_window": "--axial-window",
    "max_radius_units": "--max-radius-units",
    "profile": "--profile",
    "slice_mm": "--slice-mm",
    "scale_mm": "--scale-mm",
    "scale_units": "--scale-units",
}
BASELINE_ARGS: dict[str, Any] = {"axis_from": "path"}
IMPROVEMENT_MARGIN_MM = (
    0.05  # R median error must drop by at least this to call a candidate better; below it is "same"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------------------------------------------------------------- datasets


@dataclass
class Dataset:
    name: str
    kind: str  # "phantom" (known diameter, expected to measure) or "negative" (not a tube; must be refused)
    ply: str
    trajectory: str | None
    diameter_mm: float
    files: dict[str, str] = field(default_factory=dict)  # path -> sha256
    note: str = ""
    pinned_at: str = ""

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.files, sort_keys=True).encode()).hexdigest()[:16]


def pin_dataset(
    name: str,
    kind: str,
    ply: Path,
    trajectory: Path | None,
    diameter_mm: float,
    note: str = "",
    extra: list[Path] | None = None,
    root: Path = DATASETS,
) -> Dataset:
    """Record the dataset's files and hashes. A dataset is pinned once; re-pinning the same name is refused."""
    if kind not in ("phantom", "negative"):
        raise ValueError("kind must be 'phantom' or 'negative'")
    out = root / f"{name}.json"
    if out.exists():
        raise FileExistsError(
            f"dataset {name!r} is already pinned at {out}; pin a new name instead of changing a pinned one"
        )
    paths = [ply, *([trajectory] if trajectory else []), *(extra or [])]
    files = {str(p.resolve()): sha256(p) for p in paths}
    ds = Dataset(
        name,
        kind,
        str(ply.resolve()),
        str(trajectory.resolve()) if trajectory else None,
        diameter_mm,
        files,
        note,
        time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    root.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ds.__dict__, indent=2), encoding="utf-8")
    return ds


def load_dataset(name: str, root: Path = DATASETS) -> Dataset:
    return Dataset(**json.loads((root / f"{name}.json").read_text(encoding="utf-8")))


def verify_dataset(ds: Dataset) -> list[str]:
    """Paths whose content no longer matches the pin (empty = intact)."""
    return [p for p, h in ds.files.items() if not Path(p).exists() or sha256(Path(p)) != h]


# ---------------------------------------------------------------- candidates


@dataclass
class Candidate:
    name: str
    args: dict[str, Any]
    compound: bool = False  # only the compatibility check may combine two changes

    def change_from(self, base: dict[str, Any]) -> list[str]:
        """Knobs whose effective value differs from the baseline; a knob the candidate does not mention keeps
        the baseline value (so {'max_radius_units': 1.2} is one change, not two)."""
        effective = {
            **base,
            **{k: v for k, v in self.args.items() if v is not False and v is not None},
        }
        return sorted(k for k in effective if effective.get(k) != base.get(k))

    def check_single(self, base: dict[str, Any] | None = None) -> str:
        """The one knob this candidate changes. Refuses unknown knobs and compound candidates."""
        base = BASELINE_ARGS if base is None else base
        unknown = sorted(set(self.args) - set(KNOBS))
        if unknown:
            raise ValueError(
                f"unknown knob(s) {unknown}; candidates may change only {sorted(KNOBS)}"
            )
        changed = self.change_from(base)
        if not changed:
            return "baseline"
        if len(changed) > 1 and not self.compound:
            raise ValueError(
                f"compound candidate changes {changed}; split it into one candidate per change"
            )
        return "+".join(changed)


def benchmark_argv(ds: Dataset, args: dict[str, Any]) -> list[str]:
    argv = [sys.executable, str(TOOL), ds.ply, "--diameter-mm", str(ds.diameter_mm)]
    if ds.trajectory:
        argv += ["--trajectory", ds.trajectory]
    for k, v in args.items():
        flag = KNOBS[k]
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif v is not None:
            argv += [flag, str(v)]
    return argv


# ---------------------------------------------------------------- running


def run_candidate(
    ds: Dataset, cand: Candidate, log: Path = LOG, runs: Path = RUNS, timeout: int = 600
) -> dict[str, Any]:
    """Run the benchmark once and append the record. Refuses if the dataset's files changed since the pin."""
    change = cand.check_single()
    drift = verify_dataset(ds)
    if drift:
        raise RuntimeError(
            f"dataset {ds.name!r} changed since it was pinned: {drift}; pin a new dataset, never edit a pinned one"
        )
    run_id = time.strftime("%Y%m%dT%H%M%S") + f"-{ds.name}-{cand.name}"
    argv = benchmark_argv(ds, cand.args)
    out_path = Path(ds.ply).with_name("phantom_benchmark.json")
    out_path.unlink(
        missing_ok=True
    )  # the benchmark writes beside the PLY; a stale file must never pass as this run's
    t0 = time.perf_counter()
    try:
        r: subprocess.CompletedProcess[str] | None = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, cwd=ROOT
        )
    except subprocess.TimeoutExpired:
        r = None
    seconds = round(time.perf_counter() - t0, 2)
    output: dict[str, Any] | None = None
    if r is not None and r.returncode in (0, 1) and out_path.exists():
        output = json.loads(out_path.read_text(encoding="utf-8"))
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{run_id}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    # exit 0 = accepted, 2 = a gate refused, 1 = ordering did not hold (output written) OR a plain error such as
    # too few R points (no output): the two are told apart by whether the benchmark wrote its result
    if r is None:
        status = f"timeout_{timeout}s"
    elif r.returncode == 0:
        status = "accepted"
    elif r.returncode == 2:
        status = "refused"
    elif r.returncode == 1 and output is not None:
        status = "ordering_failed"
    else:
        status = f"error_{r.returncode}"
    stderr = (r.stderr if r is not None else "").strip()
    last_err = stderr.splitlines()[-1] if stderr else f"benchmark did not finish within {timeout} s"
    refusal = last_err if status == "refused" or status.startswith(("error", "timeout")) else ""
    blind = (output or {}).get("blind_analysis") or _freeze_status()
    classes = (output or {}).get("classes", {})
    rec = {
        "run_id": run_id,
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": ds.name,
        "dataset_kind": ds.kind,
        "dataset_digest": ds.digest,
        "candidate": cand.name,
        "args": cand.args,
        "change": change,
        "compound": cand.compound,
        "commit": git_commit(),
        "policy": blind.get("status"),
        "policy_digest": (blind.get("digest") or "")[:16],
        "status": status,
        "refusal": refusal[:300],
        "seconds": seconds,
        "metrics": {
            "fitted_R_radius_mm": (output or {}).get("fitted_R_radius_mm"),
            "ordering": (output or {}).get("ordering_R<I<U"),
            **{
                f"{k}_median_mm": classes.get(k, {}).get("median_abs_err_mm")
                for k in ("R", "I", "U")
            },
            **{f"{k}_n": classes.get(k, {}).get("n") for k in ("R", "I", "U")},
        },
        "argv": argv[2:],
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def _freeze_status() -> dict[str, Any]:
    try:
        from eczis.lab.policy_freeze import check

        return check()
    except Exception:  # noqa: BLE001 - the status is recorded, never required
        return {"status": "UNKNOWN"}


def read_log(log: Path = LOG) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    return [
        json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# ---------------------------------------------------------------- scoring (separate decision layer)


def comparable(rec: dict[str, Any], other: dict[str, Any]) -> str:
    """Why two accepted runs may NOT be compared ("" = they may). Gates: both BLIND, same policy digest (this
    also fingerprints the benchmark code), same pinned dataset, strict ordering held in both."""
    for x, who in ((rec, "candidate"), (other, "baseline")):
        if x.get("policy") != "BLIND":
            return f"{who} policy was {x.get('policy')}, not BLIND"
        if x.get("metrics", {}).get("ordering") is not True:
            return f"{who} run did not hold the strict R<I<U ordering"
    if rec.get("policy_digest") != other.get("policy_digest"):
        return "policy/benchmark code differs between the runs (freeze digests differ)"
    if rec.get("dataset_digest") != other.get("dataset_digest"):
        return "the runs used different pinned datasets"
    return ""


def score(rec: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    """Verdict for one run against the baseline run on the same dataset.

    negative dataset: the candidate must NOT accept it. accepted -> CONTROL_BROKEN (candidate rejected outright).
    phantom dataset: better / same / worse on the R median physical error with a fixed margin, and only when the
    strict ordering held; refused or unordered -> not comparable.
    """
    if rec["dataset_kind"] == "negative":
        if rec["status"] == "accepted":
            return {
                "verdict": "CONTROL_BROKEN",
                "why": "a non-tube dataset was accepted; this candidate cannot be used",
            }
        return {"verdict": "controls_hold", "why": f"negative control {rec['status']}"}
    if rec["status"] != "accepted":
        return {
            "verdict": "not_comparable",
            "why": f"{rec['status']}: {rec['refusal'] or 'ordering did not hold'}",
        }
    if baseline is None or baseline["status"] != "accepted":
        return {
            "verdict": "accepted_no_baseline",
            "why": "baseline absent or not accepted; nothing to compare against",
        }
    blocked = comparable(rec, baseline)
    if blocked:
        return {"verdict": "not_comparable", "why": blocked}
    r, b = rec["metrics"]["R_median_mm"], baseline["metrics"]["R_median_mm"]
    if r is None or b is None:
        return {"verdict": "not_comparable", "why": "R median missing"}
    d = round(b - r, 4)
    if d >= IMPROVEMENT_MARGIN_MM:
        v = "better"
    elif d <= -IMPROVEMENT_MARGIN_MM:
        v = "worse"
    else:
        v = "same"
    return {
        "verdict": v,
        "why": f"R median {r} vs baseline {b} mm (gain {d} mm, margin {IMPROVEMENT_MARGIN_MM})",
    }


def baseline_record(ds_name: str, log: Path = LOG) -> dict[str, Any] | None:
    recs = [r for r in read_log(log) if r["dataset"] == ds_name and r["change"] == "baseline"]
    return recs[-1] if recs else None


def combined(a: Candidate, b: Candidate, name: str | None = None) -> Candidate:
    """The compatibility candidate: both changes together. Only for two single-change candidates on different knobs."""
    ka, kb = a.check_single(), b.check_single()
    if "baseline" in (ka, kb) or ka == kb:
        raise ValueError(
            f"compatibility needs two single changes on different knobs, got {ka!r} and {kb!r}"
        )
    changes = {
        k: c.args[k] for c in (a, b) for k in c.change_from(BASELINE_ARGS)
    }  # only what each one changed
    return Candidate(name or f"{a.name}+{b.name}", {**BASELINE_ARGS, **changes}, compound=True)


def compatibility_verdict(
    a_rec: dict[str, Any], b_rec: dict[str, Any], ab_rec: dict[str, Any]
) -> dict[str, Any]:
    """Two winners are compatible only if together they are at least as good as the better of the two."""
    if ab_rec["status"] != "accepted":
        return {
            "verdict": "incompatible",
            "why": f"combined run {ab_rec['status']}: {ab_rec['refusal']}",
        }
    singles = [
        x
        for x in (a_rec, b_rec)
        if x["status"] == "accepted" and x["metrics"]["R_median_mm"] is not None
    ]
    if not singles:
        return {
            "verdict": "incompatible",
            "why": "neither single-change run was accepted; nothing to combine",
        }
    for x in singles:
        blocked = comparable(ab_rec, x)
        if blocked:
            return {"verdict": "incompatible", "why": blocked}
    best = min(x["metrics"]["R_median_mm"] for x in singles)
    both = ab_rec["metrics"]["R_median_mm"]
    if both is None:
        return {"verdict": "incompatible", "why": "combined run has no R median"}
    ok = (
        both <= best + IMPROVEMENT_MARGIN_MM
    )  # "at least as good" within the same margin the scorer uses
    return {
        "verdict": "compatible" if ok else "incompatible",
        "why": f"combined R median {both} vs best single {best} mm (margin {IMPROVEMENT_MARGIN_MM})",
    }


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def report(log: Path = LOG) -> str:
    recs = read_log(log)
    if not recs:
        return "no experiments logged"
    lines = [
        f"{'run':42s} {'change':18s} {'status':16s} {'policy':8s} {'R_med':>7s} {'I_med':>7s} {'U_med':>7s}  verdict"
    ]
    for r in recs:
        base = baseline_record(r["dataset"], log) if r["change"] != "baseline" else None
        v = (
            score(r, base)["verdict"]
            if r["change"] != "baseline"
            else ("accepted" if r["status"] == "accepted" else r["status"])
        )
        m = r["metrics"]
        lines.append(
            f"{r['run_id'][:42]:42s} {r['change'][:18]:18s} {r['status'][:16]:16s} {str(r['policy'])[:8]:8s} "
            f"{_fmt(m.get('R_median_mm')):>7s} {_fmt(m.get('I_median_mm')):>7s} {_fmt(m.get('U_median_mm')):>7s}  {v}"
        )
        if r["refusal"]:
            lines.append(f"{'':42s} refused: {r['refusal'][:110]}")
    return "\n".join(lines)
