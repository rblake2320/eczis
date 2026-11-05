# UNFILED-IP
"""T1: schemas, canonical serialization, and fail-closed loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from eczis.bench.pipeline import CaseResult
from eczis.config import DemoConfig, load_config
from eczis.schemas import CaptureSet
from eczis.seal.canonical import canonical_json
from eczis.storage import load_capture, save_capture
from eczis.versioning import SchemaVersionError, check_schema_version


def _digest(capture: CaptureSet) -> str:
    h = hashlib.sha256()
    for frame in capture.frames:
        h.update(frame.data.tobytes())
        h.update(frame.valid_mask.tobytes())
    return h.hexdigest()


def _rewrite_metadata(src: Path, dst: Path, edit: object) -> Path:
    with np.load(src, allow_pickle=False) as bundle:
        arrays = {k: bundle[k] for k in bundle.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    edit(metadata)  # type: ignore[operator]
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(dst, **arrays)
    return dst


def test_capture_round_trip_preserves_values_and_hashes(coin_case: CaseResult, tmp_path: Path) -> None:
    path = save_capture(coin_case.capture, tmp_path / "capture.npz")
    loaded = load_capture(path)
    assert _digest(loaded) == _digest(coin_case.capture)
    assert loaded.calibration == coin_case.capture.calibration
    assert [f.params for f in loaded.frames] == [f.params for f in coin_case.capture.frames]


def test_unknown_major_version_fails_closed(coin_case: CaseResult, tmp_path: Path) -> None:
    src = save_capture(coin_case.capture, tmp_path / "capture.npz")

    def bump(meta: dict[str, object]) -> None:
        meta["schema_version"] = "2.0"

    with pytest.raises(SchemaVersionError, match="unsupported schema major 2"):
        load_capture(_rewrite_metadata(src, tmp_path / "v2.npz", bump))


def test_unknown_observation_field_fails_closed(coin_case: CaseResult, tmp_path: Path) -> None:
    src = save_capture(coin_case.capture, tmp_path / "capture.npz")

    def inject(meta: dict[str, list[dict[str, object]]]) -> None:
        meta["observations"][0]["unexpected_field"] = 1

    with pytest.raises(TypeError):
        load_capture(_rewrite_metadata(src, tmp_path / "extra.npz", inject))


def test_frame_count_mismatch_fails_closed(coin_case: CaseResult, tmp_path: Path) -> None:
    src = save_capture(coin_case.capture, tmp_path / "capture.npz")

    def drop(meta: dict[str, list[object]]) -> None:
        meta["observations"].pop()

    with pytest.raises(ValueError, match="observation count"):
        load_capture(_rewrite_metadata(src, tmp_path / "drop.npz", drop))


def test_invalid_frames_fail_closed(coin_case: CaseResult) -> None:
    frame = coin_case.capture.frames[0]
    with pytest.raises(ValueError, match="masks must match"):
        replace(frame, valid_mask=frame.valid_mask[:-1]).validate()
    with pytest.raises(TypeError, match="float32"):
        replace(frame, data=frame.data.astype(np.float64)).validate()
    with pytest.raises(ValueError, match="Saturated"):
        replace(frame, saturation_mask=frame.valid_mask.copy()).validate()
    duplicate = replace(coin_case.capture, frames=[frame, frame])
    with pytest.raises(ValueError, match="unique"):
        duplicate.validate()


@pytest.mark.parametrize("version", [None, "", "x.1", 1, "3.0"])
def test_schema_version_policy(version: object) -> None:
    with pytest.raises(SchemaVersionError):
        check_schema_version(version, kind="test")
    assert check_schema_version("1.7", kind="test") == "1.7"


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})
    assert canonical_json({"a": 1}) == b'{"a":1}'


def test_config_loader_fails_closed(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"sensor_height": 24, "frame_count": 9}), encoding="utf-8")
    assert load_config(good, root_seed=5).root_seed == 5
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"sensor_heigth": 24}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(bad)
    with pytest.raises(ValueError, match="square"):
        DemoConfig(frame_count=10)
