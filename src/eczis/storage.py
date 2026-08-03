# UNFILED-IP
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eczis.schemas import Calibration, CaptureSet, ObservationParams, RawFrame


def save_capture(capture: CaptureSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.stack([frame.data for frame in capture.frames])
    valid = np.stack([frame.valid_mask for frame in capture.frames])
    saturation = np.stack([frame.saturation_mask for frame in capture.frames])
    metadata = {
        "schema_version": capture.schema_version,
        "target_id": capture.target_id,
        "root_seed": capture.root_seed,
        "calibration": asdict(capture.calibration),
        "truth_digest": capture.truth_digest,
        "metadata": capture.metadata,
        "observations": [asdict(frame.params) for frame in capture.frames],
    }
    np.savez_compressed(
        path,
        data=data.astype(np.float32),
        valid=valid.astype(np.bool_),
        saturation=saturation.astype(np.bool_),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    return path


def load_capture(path: Path) -> CaptureSet:
    with np.load(path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata_json"].item()))
        calibration_data = metadata["calibration"]
        calibration_data["sensor_shape"] = tuple(calibration_data["sensor_shape"])
        calibration = Calibration(**calibration_data)
        data = bundle["data"].astype(np.float32)
        valid = bundle["valid"].astype(np.bool_)
        saturation = bundle["saturation"].astype(np.bool_)
    observations = [ObservationParams(**item) for item in metadata["observations"]]
    frames = [
        RawFrame(data[i], valid[i], saturation[i], observations[i])
        for i in range(len(observations))
    ]
    capture = CaptureSet(
        schema_version=metadata["schema_version"],
        target_id=metadata["target_id"],
        root_seed=int(metadata["root_seed"]),
        calibration=calibration,
        frames=frames,
        truth_digest=metadata["truth_digest"],
        metadata=metadata.get("metadata", {}),
    )
    capture.validate()
    return capture
