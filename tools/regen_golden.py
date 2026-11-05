# UNFILED-IP
"""Regenerate tests/golden/golden_values.json.

Run only for a deliberate, reviewed change to targets, schedules, manifest
format, or numerics -- and record why in CHANGELOG.md. Golden drift without a
decision record is a regression by definition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def compute() -> dict[str, Any]:
    import hashlib
    import tempfile
    from dataclasses import asdict

    import numpy as np
    from golden_support import GOLDEN_CFG, fixture_tree

    from eczis.bench.pipeline import make_target, run_case
    from eczis.rig.schedule import lattice_schedule
    from eczis.seal.canonical import canonical_json
    from eczis.seal.manifest import build_manifest

    schedule = [asdict(o) for o in lattice_schedule(16, 20260803).observations]
    targets: dict[str, Any] = {}
    for family in ("microglyph", "siemens", "coin", "iris", "pcb"):
        t = make_target(family, GOLDEN_CFG.hr_shape, GOLDEN_CFG.root_seed)
        h, w = t.image.shape
        blocks = t.image.reshape(4, h // 4, 4, w // 4).mean(axis=(1, 3))
        targets[family] = {
            "mean": float(t.image.mean()),
            "std": float(t.image.std()),
            "feature_pixels": int(t.feature_mask.sum()),
            "block_means": [[float(v) for v in row] for row in blocks],
        }
    with tempfile.TemporaryDirectory() as tmp:
        root = fixture_tree(Path(tmp))
        manifest = build_manifest(root, environment={"fixed": True})
    case = run_case("coin", GOLDEN_CFG).metrics
    return {
        "schedule_sha256": hashlib.sha256(canonical_json(schedule)).hexdigest(),
        "manifest_sha256": hashlib.sha256(canonical_json(manifest)).hexdigest(),
        "targets": targets,
        "coin_case": {
            "map_psnr_db": case["methods"]["map"]["psnr_db"],
            "shift_add_psnr_db": case["methods"]["shift_add"]["psnr_db"],
            "r_fraction": case["provenance"]["r_fraction"],
            "false_r_rate": case["provenance"]["false_r_rate"],
            "gate_pass_fraction": case["gate"]["base_pass_fraction"],
        },
        "_numpy_used": np.__version__,
    }


def main() -> int:
    out = ROOT / "tests" / "golden" / "golden_values.json"
    out.write_text(json.dumps(compute(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
