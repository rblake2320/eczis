# UNFILED-IP
"""T9: staged CLI workflow, resumption safety, and fail-closed behaviour."""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from eczis.cli import main
from eczis.config import DemoConfig


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    out = capsys.readouterr().out
    return int(exit_info.value.code or 0), json.loads(out) if out.strip().startswith("{") else {}


@pytest.fixture()
def cfg_file(tmp_path: Path, small_cfg: DemoConfig) -> Path:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(asdict(small_cfg)), encoding="utf-8")
    return path


def test_doctor(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(["doctor"], capsys)
    assert code == 0 and out["models_are_distinct"] is True


def test_staged_pipeline_end_to_end(tmp_path: Path, cfg_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    t, c, r, g, p, v = (tmp_path / n for n in ("target", "capture.npz", "recon", "gate", "prov", "viewer.html"))
    assert run(["target", "create", "--output", str(t), "--family", "coin", "--config", str(cfg_file)], capsys)[0] == 0
    assert run(["capture", "simulate", "--target", str(t), "--output", str(c), "--frames", "9",
                "--config", str(cfg_file)], capsys)[0] == 0
    assert run(["reconstruct", "--capture", str(c), "--output", str(r), "--iterations", "6"], capsys)[0] == 0
    assert run(["gate", "--capture", str(c), "--reconstruction", str(r / "map.npy"), "--output", str(g)], capsys)[0] == 0
    assert run(["provenance", "--capture", str(c), "--reconstruction", str(r / "map.npy"), "--gate", str(g),
                "--output", str(p)], capsys)[0] == 0
    assert run(["render", "--reconstruction", str(r / "map.npy"), "--provenance", str(p), "--gate", str(g),
                "--output", str(v)], capsys)[0] == 0
    bundle = tmp_path / "staged.eczis"
    assert run(["seal", "--input", str(p), "--output", str(bundle)], capsys)[0] == 0
    code, out = run(["verify", "--bundle", str(bundle)], capsys)
    assert code == 0 and out["valid"] is True


def test_missing_input_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(["reconstruct", "--capture", str(tmp_path / "nope.npz"), "--output", str(tmp_path / "o")], capsys)
    assert code == 2 and out["status"] == "FAIL"


def test_unknown_config_key_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"not_a_key": 1}', encoding="utf-8")
    code, out = run(["target", "create", "--output", str(tmp_path / "t"), "--config", str(bad)], capsys)
    assert code == 2 and "unknown config keys" in out["error"]


def test_export_verifies_then_extracts(demo_run: tuple[Path, dict[str, Any]], tmp_path: Path,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    output, result = demo_run
    bundle = output / "eczis_demo.eczis"
    code, out = run(["export", "--bundle", str(bundle), "--output", str(tmp_path / "x"),
                     "--seed", str(result["root_seed"])], capsys)
    assert code == 0 and "viewer.html" in out["exported"]
    tampered = tmp_path / "tampered.eczis"
    with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(tampered, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            dst.writestr(item.filename, data + b"!" if item.filename == "viewer.html" else data)
    code, out = run(["export", "--bundle", str(tampered), "--output", str(tmp_path / "y"),
                     "--seed", str(result["root_seed"])], capsys)
    assert code == 3 and out["status"] == "FAIL"
    assert not (tmp_path / "y" / "viewer.html").exists()


def test_verify_with_key_file_and_wrong_key(demo_run: tuple[Path, dict[str, Any]], tmp_path: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    output, _ = demo_run
    code, _ = run(["verify", "--bundle", str(output / "eczis_demo.eczis"), "--key", str(output / "DEMO_ONLY.key")], capsys)
    assert code == 0
    code, out = run(["verify", "--bundle", str(output / "eczis_demo.eczis"), "--seed", "1"], capsys)
    assert code == 3 and out["valid"] is False


def test_key_create_refuses_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    key = tmp_path / "seal.key"
    assert run(["key", "create", "--output", str(key), "--protect", "none"], capsys)[0] == 0
    code, out = run(["key", "create", "--output", str(key), "--protect", "none"], capsys)
    assert code == 2 and "overwrite" in out["reason"]


def test_bench_command(tmp_path: Path, cfg_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(["bench", "--output", str(tmp_path / "b"), "--split", "development", "--family", "coin",
                     "--config", str(cfg_file), "--no-mismatch"], capsys)
    assert code == 0 and out["cases_per_arm"] == 3
    assert out["acceptance"]["max_injected_pixels_labeled_r_or_m"] == 0
