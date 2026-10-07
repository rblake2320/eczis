# UNFILED-IP
"""Dense-cloud evidence grading: visibility and angular spread must separate R / I / U on a scene
where some points are seen by one camera only, and PLY/TUM readers must round-trip."""
from __future__ import annotations

import numpy as np

from eczis.lab.grade_dense import (
    Pose,
    grade_points,
    read_ply,
    read_tum_trajectory,
    write_graded_ply,
)

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])


def _rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def test_visibility_and_spread_separate_classes() -> None:
    # four cameras on an arc looking at the origin region; plus one far-off point seen by one camera only
    poses = [Pose(float(i), np.array([np.sin(np.radians(a)) * 5, 0.0, 5 - np.cos(np.radians(a)) * 5]), _rot_y(-a))
             for i, a in enumerate((-12, -4, 4, 12))]
    wall = np.random.default_rng(0).uniform([-0.5, -0.5, 4.8], [0.5, 0.5, 5.2], (300, 3))  # seen by all
    side = np.array([[4.0, 0.0, 1.0]])  # only in the last camera's view (and ahead of it)
    behind = np.array([[0.0, 0.0, -3.0]])  # behind every camera
    g = grade_points(np.vstack([wall, side, behind]), poses, K, (640, 480))
    assert g["counts"]["R"] >= 250 and g["spread_deg"][:300].min() > 2.0
    assert g["classes"][300] in ("U", "I") and g["views"][300] <= 2
    assert g["classes"][301] == "U" and g["views"][301] == 0


def test_two_views_same_direction_is_I_not_R() -> None:
    # two cameras at the same spot: 2 views but ~0 deg spread -> I (seen twice, not triangulable)
    p = Pose(0.0, np.zeros(3), np.eye(3))
    g = grade_points(np.array([[0.0, 0.0, 3.0]]), [p, Pose(1.0, np.zeros(3), np.eye(3)), Pose(2.0, np.zeros(3), np.eye(3))], K, (640, 480))
    assert g["views"][0] == 3 and g["spread_deg"][0] < 0.01 and g["classes"][0] == "I"


def test_ply_and_tum_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    xyz = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
    write_graded_ply(tmp_path / "g.ply", xyz, np.array(["R", "U"]), np.array([5, 0], dtype=np.int32))
    back, rgb = read_ply(tmp_path / "g.ply")
    assert np.allclose(back, xyz, atol=1e-5) and rgb is not None and tuple(rgb[0]) == (37, 99, 235)
    (tmp_path / "t.txt").write_text("0.0 1 2 3 0 0 0 1\n1.0 0 0 0 0 0.7071068 0 0.7071068\n", encoding="utf-8")
    poses = read_tum_trajectory(tmp_path / "t.txt")
    assert len(poses) == 2 and np.allclose(poses[0].position, [1, 2, 3]) and np.allclose(poses[0].rot_wc, np.eye(3))
    assert np.allclose(poses[1].rot_wc @ np.array([0, 0, 1.0]), [1, 0, 0], atol=1e-6)  # 90 deg about y maps z->x
