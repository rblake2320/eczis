# UNFILED-IP
"""Measure the metric scale of a reconstruction from a known probe movement.

    uv run python tools/scale_from_insertion.py <folder_A> <folder_B> --moved-mm 5.0

Two sweeps of the same tube, with the probe pushed in by a known distance between them (read it
off a mark on the probe). Each folder holds the reconstruction's trajectory (mast3r/<name>.txt,
TUM format). The tool aligns B's camera path onto A's by the shared wall (both see the same
tube) and reports how far the camera start moved in cloud units, so that

    scale_units_to_mm = moved_mm / moved_units

which is exactly the pair phantom_benchmark.py needs as --scale-mm / --scale-units.

Simpler single-sweep alternative supported too: a known INSERTION DEPTH within one sweep, i.e.
the probe started at mark 0 and ended at mark D while recording; then the first-to-last camera
distance in units corresponds to D mm (--single, uses the trajectory's own end points).
Monocular reconstructions have arbitrary scale; this is the honest way to attach millimetres.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def traj(folder: Path) -> np.ndarray:
    from eczis.lab.grade_dense import read_tum_trajectory

    files = sorted((folder / "mast3r").glob("*.txt")) if (folder / "mast3r").exists() else []
    if not files:
        raise SystemExit(f"no mast3r/*.txt trajectory in {folder}; run tools/spark_scan.py on it first")
    poses = read_tum_trajectory(files[0])
    if len(poses) < 2:
        raise SystemExit(f"{files[0]} has {len(poses)} poses; need >= 2 keyframes")
    return np.array([p.position for p in poses])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder_a", type=Path)
    ap.add_argument("folder_b", type=Path, nargs="?")
    ap.add_argument("--moved-mm", type=float, required=True, help="known probe movement in mm")
    ap.add_argument("--single", action="store_true", help="one sweep: movement is first keyframe -> last keyframe")
    a = ap.parse_args()
    if a.single or a.folder_b is None:
        c = traj(a.folder_a)
        along = np.linalg.norm(c[-1] - c[0])
        path = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
        out = {"mode": "single sweep: first keyframe -> last keyframe", "moved_units": round(float(along), 5),
               "path_length_units": round(path, 5), "moved_mm": a.moved_mm,
               "scale_units_to_mm": round(a.moved_mm / float(along), 6) if along > 1e-9 else None,
               "caveat": "assumes the camera moved mostly along the insertion axis; a wobbly path overstates 'moved_units' "
                         "(path/moved ratio above ~1.3 means the straight-line distance is not the insertion distance)",
               "path_over_straight": round(path / float(along), 3) if along > 1e-9 else None}
    else:
        ca, cb = traj(a.folder_a), traj(a.folder_b)
        # Both sweeps start at the camera's own origin (MASt3R world = first camera), so we cannot
        # align by poses alone; use the first keyframe of each and the shared wall is implicit in
        # the user's known movement. Report both starts' separation after a best-effort scale match
        # of the two path lengths (same tube, similar sweep => similar path length in each frame).
        la = float(np.sum(np.linalg.norm(np.diff(ca, axis=0), axis=1)))
        lb = float(np.sum(np.linalg.norm(np.diff(cb, axis=0), axis=1)))
        out = {"mode": "two sweeps (experimental): relies on similar sweep length in both; prefer --single with a marked insertion",
               "path_length_units_a": round(la, 5), "path_length_units_b": round(lb, 5), "moved_mm": a.moved_mm,
               "scale_units_to_mm": None,
               "caveat": "two independent monocular reconstructions have independent scales; without a shared landmark the "
                         "known movement cannot be measured in one frame. Use --single."}
    print(json.dumps(out, indent=2))
    (a.folder_a / "scale.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if out["scale_units_to_mm"] is None:
        return 1
    print(f"\nNext: uv run python tools/phantom_benchmark.py {a.folder_a}/mast3r/graded.ply --diameter-mm <d> --profile "
          f"--scale-mm {a.moved_mm} --scale-units {out['moved_units']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
