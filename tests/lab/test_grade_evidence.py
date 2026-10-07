# UNFILED-IP
"""Evidence grader on synthetic keyframes with known geometry.

Scene note: cameras at +/-3 deg, +/-0.3 units (at +/-8 deg only 21% of one keyframe's wall fell inside
another's image, which tested field of view rather than grading).
Pre-registered: (a) a consistent wall seen by 3 cameras with real parallax is R; (b) the same
wall seen by one camera only is never R; (c) a drifted sheet (duplicate surface offset toward
the cameras) is demoted to U by the depth-conflict rule; (d) removing views cannot strengthen
a point; (e) duplicate cameras (zero parallax) cannot make R; (f) post-loss keyframes cap at I.
These run without MASt3R; the real export is validated separately on the simulated canal.
"""
from __future__ import annotations

import numpy as np
import pytest

from eczis.lab.grade_evidence import POLICY, Keyframe, grade_keyframes

K = np.array([[200.0, 0, 80], [0, 200.0, 60], [0, 0, 1.0]])
H, W = 120, 160


def _rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _wall_points(rot: np.ndarray, pos: np.ndarray, depth_offset: float = 0.0, z_wall: float = 6.0) -> np.ndarray:
    """Points of the plane z = z_wall (world) seen through pixel grid of camera (rot, pos)."""
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    rays_w = rays_c @ rot.T
    t = (z_wall - pos[2]) / rays_w[..., 2]
    pw = pos + rays_w * t[..., None]
    if depth_offset:
        pw = pw + (pos - pw) * depth_offset  # pull toward the camera: a drifted / floating sheet
    return pw


def _texture(pw: np.ndarray) -> np.ndarray:
    return 0.5 + 0.3 * np.sin(3.1 * pw[..., 0]) * np.cos(2.7 * pw[..., 1])


def _kf(i: int, deg: float, x: float, frame_id: int | None = None, drift: float = 0.0) -> Keyframe:
    rot, pos = _rot_y(deg), np.array([x, 0.0, 0.0])
    pw = _wall_points(rot, pos, drift)
    true_pw = _wall_points(rot, pos, 0.0)
    return Keyframe(i, frame_id if frame_id is not None else i, float(i), rot, pos, 1.0, pw,
                    np.ones((H, W), np.float32), _texture(true_pw), K)


def _counts(kfs: list[Keyframe], loss: int | None = None) -> dict[str, float]:
    _, _, g = grade_keyframes(kfs, loss_after_frame=loss, stride=4)
    n = len(g.classes)
    return {c: float((g.classes == c).mean()) for c in "RIU"} | {"n": n}


def test_consistent_wall_with_parallax_is_mostly_R() -> None:
    c = _counts([_kf(0, -3, -0.3), _kf(1, 0, 0.0), _kf(2, 3, 0.3)])
    assert c["R"] > 0.6, c


def test_single_view_is_never_R() -> None:
    c = _counts([_kf(0, 0, 0.0)])
    assert c["R"] == 0.0 and c["U"] == 1.0


def test_duplicate_cameras_cannot_make_R() -> None:
    c = _counts([_kf(0, 0, 0.0), _kf(1, 0, 0.0), _kf(2, 0, 0.0)])  # three identical viewpoints
    assert c["R"] == 0.0 and c["I"] > 0.9, c  # agree in depth+colour but zero parallax


def test_drifted_sheet_is_demoted_by_depth_conflict() -> None:
    good = [_kf(0, -3, -0.3), _kf(1, 0, 0.0), _kf(2, 3, 0.3)]
    base = _counts(good)
    drifted = good + [_kf(3, 1.5, 0.15, drift=0.3)]  # a fourth keyframe whose surface floats 30% nearer
    _, _, g = grade_keyframes(drifted, stride=4)
    per_kf = len(g.classes) // 4
    sheet = g.classes[3 * per_kf :]
    assert (sheet == "R").mean() < 0.05, "floating sheet must not be R"
    assert (sheet == "U").mean() > 0.5, "floating sheet should mostly be U (conflict / no agreement)"
    assert base["R"] > 0.6  # the real wall keeps its support


def test_removing_views_cannot_strengthen() -> None:
    three = _counts([_kf(0, -3, -0.3), _kf(1, 0, 0.0), _kf(2, 3, 0.3)])
    two = _counts([_kf(0, -3, -0.3), _kf(1, 0, 0.0)])
    assert two["R"] <= three["R"] + 1e-9 and two["R"] == 0.0  # R needs 2 OTHER agreeing views


def test_post_loss_keyframes_cap_at_I() -> None:
    kfs = [_kf(0, -3, -0.3, frame_id=0), _kf(1, 0, 0.0, frame_id=40), _kf(2, 3, 0.3, frame_id=80), _kf(3, 1.5, 0.15, frame_id=95)]
    _, _, g = grade_keyframes(kfs, loss_after_frame=90, stride=4)
    per_kf = len(g.classes) // 4
    assert (g.classes[3 * per_kf :] == "R").sum() == 0 and g.reasons["post_loss_capped"] > 0


def test_policy_is_versioned() -> None:
    assert POLICY["version"].startswith("grade_evidence/") and POLICY["r_min_agree"] >= 2


@pytest.mark.parametrize("tol", [0.0])
def test_zero_colour_tolerance_demotes_almost_everything(tol: float) -> None:
    """Exact-intensity matches survive nearest-pixel lookup only rarely, so R drops from ~0.7 to < 0.2."""
    pol = {**POLICY, "colour_tol": tol}
    _, _, g = grade_keyframes([_kf(0, -3, -0.3), _kf(1, 0, 0.0), _kf(2, 3, 0.3)], policy=pol, stride=4)
    assert (g.classes == "R").mean() < 0.2  # bilinear-free nearest-pixel lookup rarely matches exactly


def test_conflict_counts_are_populated_per_point() -> None:
    """Grade.conflict must carry the per-point conflict counts, not zeros, and agree with the U demotions."""
    kfs = [_kf(0, -3, -0.3), _kf(1, 0, 0.0), _kf(2, 3, 0.3), _kf(3, 1.5, 0.15, drift=0.3)]  # last one floats nearer
    _, _, g = grade_keyframes(kfs, stride=4)
    assert g.conflict.shape == g.classes.shape
    assert g.reasons["depth_conflict_sheet"] == int((g.conflict > 0).sum()) > 0
    assert (g.classes[g.conflict > 0] == "U").all()
