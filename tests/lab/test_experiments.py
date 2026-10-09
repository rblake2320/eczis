# UNFILED-IP
"""Experiment runner: pinned datasets, one change per candidate, losers kept, controls on every candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eczis.lab import experiments as ex

RGB = {"R": (37, 99, 235), "I": (245, 158, 11), "U": (120, 120, 120)}


def _cloud(
    length: float, theta_lo: float, theta_hi: float, r: float = 3.5, n: int = 2000, seed: int = 0
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    pts, cls = [], []
    for k, noise in (("R", 0.05), ("I", 0.3), ("U", 0.8)):
        t = rng.uniform(np.radians(theta_lo), np.radians(theta_hi), n)
        z = rng.uniform(0, length, n)
        rr = r + rng.normal(0, noise, n)
        pts.append(np.stack([rr * np.cos(t), rr * np.sin(t), z], axis=1))
        cls += [k] * n
    return np.vstack(pts), cls


def _write(tmp: Path, xyz: np.ndarray, cls: list[str], name: str = "graded.ply") -> Path:
    lines = [
        f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {RGB[c][0]} {RGB[c][1]} {RGB[c][2]}"
        for p, c in zip(xyz, cls, strict=True)
    ]
    header = (
        f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path = tmp / name
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def _traj(tmp: Path, length: float) -> Path:
    p = tmp / "traj.txt"
    p.write_text(
        "\n".join(
            f"{i} 0.1 -0.2 {z:.3f} 0 0 0 1" for i, z in enumerate(np.linspace(0.5, length - 0.5, 8))
        )
        + "\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """A tube phantom and a not-a-tube negative control, each pinned in a temp experiments root."""
    monkeypatch.setattr(ex, "DATASETS", tmp_path / "datasets")
    monkeypatch.setattr(ex, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(ex, "LOG", tmp_path / "log.jsonl")
    d1 = tmp_path / "tube"
    d1.mkdir()
    xyz, cls = _cloud(3.0, -70, 70)
    tube = ex.pin_dataset(
        "tube7",
        "phantom",
        _write(d1, xyz, cls),
        _traj(d1, 3.0),
        7.0,
        "synthetic half-wrapped tube",
        root=ex.DATASETS,
    )
    d2 = tmp_path / "notube"
    d2.mkdir()
    xyz2, cls2 = _cloud(3.0, -70, 70, seed=1)
    notube = ex.pin_dataset(
        "notube",
        "negative",
        _write(d2, xyz2, cls2),
        None,
        7.0,
        "no trajectory: PCA must refuse",
        root=ex.DATASETS,
    )
    return {"tube": tube, "notube": notube, "tmp": tmp_path}


def _run(ds: ex.Dataset, cand: ex.Candidate) -> dict[str, object]:
    return ex.run_candidate(ds, cand, log=ex.LOG, runs=ex.RUNS)


def test_pin_is_once_and_drift_is_refused(lab: dict[str, object]) -> None:
    tube: ex.Dataset = lab["tube"]  # type: ignore[assignment]
    assert len(tube.files) == 2 and ex.verify_dataset(tube) == []
    with pytest.raises(FileExistsError):
        ex.pin_dataset("tube7", "phantom", Path(tube.ply), None, 7.0, root=ex.DATASETS)
    Path(tube.ply).write_text(
        Path(tube.ply).read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )  # someone edits the pinned file
    assert ex.verify_dataset(tube) == [tube.ply]
    with pytest.raises(RuntimeError, match="changed since it was pinned"):
        _run(tube, ex.Candidate("baseline", dict(ex.BASELINE_ARGS)))
    assert not ex.LOG.exists()  # a refused run is not a run


def test_one_change_per_candidate() -> None:
    assert ex.Candidate("b", dict(ex.BASELINE_ARGS)).check_single() == "baseline"
    assert ex.Candidate("o", {"axis_from": "optical"}).check_single() == "axis_from"
    with pytest.raises(ValueError, match="compound"):
        ex.Candidate("two", {"axis_from": "optical", "profile": True}).check_single()
    with pytest.raises(ValueError, match="unknown knob"):
        ex.Candidate(
            "bad", {"depth_rel_tol": 0.5}
        ).check_single()  # a policy constant is not a knob
    both = ex.combined(
        ex.Candidate("o", {"axis_from": "optical"}),
        ex.Candidate("p", {"axis_from": "path", "profile": True}),
    )
    assert both.compound and both.check_single() == "axis_from+profile"
    with pytest.raises(ValueError, match="different knobs"):
        ex.combined(
            ex.Candidate("o", {"axis_from": "optical"}),
            ex.Candidate("f", {"axis_from": "path+fit"}),
        )


def test_baseline_candidate_and_controls_end_to_end(lab: dict[str, object]) -> None:
    tube: ex.Dataset = lab["tube"]  # type: ignore[assignment]
    notube: ex.Dataset = lab["notube"]  # type: ignore[assignment]
    base = _run(tube, ex.Candidate("baseline", dict(ex.BASELINE_ARGS)))
    assert (
        base["status"] == "accepted"
        and base["metrics"]["ordering"] is True
        and base["change"] == "baseline"
    )
    assert abs(base["metrics"]["fitted_R_radius_mm"] - 3.5) < 0.05
    assert base["policy"] in ("BLIND", "CHANGED", "NOT_FROZEN") and base["commit"]
    fit = _run(tube, ex.Candidate("fit", {"axis_from": "path+fit"}))
    assert fit["status"] == "accepted" and fit["change"] == "axis_from"
    v = ex.score(fit, base)
    assert v["verdict"] in ("same", "better", "worse") and "margin" in v["why"]
    # the negative control runs on the same candidate and must be refused
    neg = _run(notube, ex.Candidate("fit", {"axis_from": "path+fit"}))
    assert neg["status"] == "refused" and "axis underdetermined" in neg["refusal"]
    assert ex.score(neg, None)["verdict"] == "controls_hold"
    # losers and refusals stay: three lines, append-only, each with the dataset digest and the declared change
    recs = ex.read_log(ex.LOG)
    assert [r["status"] for r in recs] == ["accepted", "accepted", "refused"]
    assert all(r["dataset_digest"] and r["change"] for r in recs)
    assert (ex.RUNS / f"{base['run_id']}.json").exists()
    assert ex.baseline_record("tube7", ex.LOG)["run_id"] == base["run_id"]
    txt = ex.report(ex.LOG)
    assert "controls_hold" in txt and "baseline" in txt and "refused:" in txt


def test_control_broken_rejects_a_candidate_outright() -> None:
    fake = {
        "dataset_kind": "negative",
        "status": "accepted",
        "refusal": "",
        "metrics": {"R_median_mm": 0.01},
    }
    assert ex.score(fake, None)["verdict"] == "CONTROL_BROKEN"


def _acc(**kw: object) -> dict[str, object]:
    base = {
        "dataset_kind": "phantom",
        "status": "accepted",
        "refusal": "",
        "policy": "BLIND",
        "policy_digest": "p1",
        "dataset_digest": "d1",
        "metrics": {"R_median_mm": 0.10, "ordering": True},
    }
    base.update(kw)
    return base


def test_compatibility_verdict_rules() -> None:
    a = _acc(metrics={"R_median_mm": 0.10, "ordering": True})
    b = _acc(metrics={"R_median_mm": 0.12, "ordering": True})
    good = _acc(metrics={"R_median_mm": 0.11, "ordering": True})
    bad = _acc(metrics={"R_median_mm": 0.30, "ordering": True})
    refused = {
        "status": "refused",
        "refusal": "cylinder fit tilted",
        "metrics": {"R_median_mm": None},
    }
    assert ex.compatibility_verdict(a, b, good)["verdict"] == "compatible"
    assert ex.compatibility_verdict(a, b, bad)["verdict"] == "incompatible"
    assert ex.compatibility_verdict(a, b, refused)["verdict"] == "incompatible"


def test_cli_parses_knobs_and_reports(lab: dict[str, object], tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "experiment_cli", ex.ROOT / "tools" / "experiment.py"
    )
    cli = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(cli)  # type: ignore[union-attr]
    assert cli.parse_kv(["axis_from=optical"]) == {"axis_from": "optical"}
    assert cli.parse_kv(["max_radius_units=1.5"]) == {"axis_from": "path", "max_radius_units": 1.5}
    assert cli.parse_kv(["profile=true"]) == {"axis_from": "path", "profile": True}
    assert (
        json.loads(json.dumps(cli.parse_kv(["axial_window=0.2:1.4"])))["axial_window"] == "0.2:1.4"
    )


def test_plain_benchmark_errors_are_not_called_ordering_failures(lab: dict[str, object]) -> None:
    """exit 1 without an output file is an error (e.g. too few R points), not 'ordering failed'; and a stale
    phantom_benchmark.json beside the PLY must never be read as this run's result."""
    tube: ex.Dataset = lab["tube"]  # type: ignore[assignment]
    stale = Path(tube.ply).with_name("phantom_benchmark.json")
    stale.write_text(
        json.dumps({"ordering_R<I<U": True, "classes": {"R": {"median_abs_err_mm": 0.0, "n": 1}}}),
        encoding="utf-8",
    )
    rec = _run(
        tube, ex.Candidate("tiny", {"max_radius_units": 0.001})
    )  # the window keeps ~no points -> 'only N R points'
    assert rec["status"] == "error_1" and "R points" in rec["refusal"]
    assert rec["metrics"]["R_median_mm"] is None  # the stale file was not picked up
    assert ex.score(rec, None)["verdict"] == "not_comparable"


def test_better_needs_blind_same_policy_same_dataset_and_ordering() -> None:
    """Review finding 2026-10-08: 'better' must not be reachable without the blind-analysis gates."""
    cand = _acc(metrics={"R_median_mm": 0.02, "ordering": True})
    assert ex.score(cand, _acc())["verdict"] == "better"
    assert (
        ex.score(_acc(policy="CHANGED", metrics={"R_median_mm": 0.02, "ordering": True}), _acc())[
            "verdict"
        ]
        == "not_comparable"
    )
    assert ex.score(cand, _acc(policy="NOT_FROZEN"))["verdict"] == "not_comparable"
    assert (
        ex.score(_acc(policy_digest="p2", metrics={"R_median_mm": 0.02, "ordering": True}), _acc())[
            "verdict"
        ]
        == "not_comparable"
    )
    assert (
        ex.score(
            _acc(dataset_digest="d2", metrics={"R_median_mm": 0.02, "ordering": True}), _acc()
        )["verdict"]
        == "not_comparable"
    )
    assert (
        ex.score(_acc(metrics={"R_median_mm": 0.02, "ordering": None}), _acc())["verdict"]
        == "not_comparable"
    )
    assert (
        ex.compatibility_verdict(_acc(), _acc(), _acc(policy="CHANGED"))["verdict"]
        == "incompatible"
    )
    refused = {"status": "refused", "refusal": "x", "metrics": {"R_median_mm": None}}
    assert (
        ex.compatibility_verdict(refused, refused, _acc())["verdict"] == "incompatible"
    )  # no empty-min crash


def test_false_knob_is_not_a_change() -> None:
    assert ex.Candidate("p", {"profile": False}).check_single() == "baseline"
    assert ex.Candidate("p", {"profile": True}).check_single() == "profile"
