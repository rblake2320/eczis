# UNFILED-IP
from __future__ import annotations

from pathlib import Path

import numpy as np


def test_challenge_changes_provenance_in_false_region(demo_run: tuple[Path, dict[str, object]]) -> None:
    output, _ = demo_run
    base = np.load(output / "artifacts" / "provenance.npy", allow_pickle=False)
    challenged = np.load(output / "artifacts" / "challenge_provenance.npy", allow_pickle=False)
    assert int((challenged < base).sum()) > 0
