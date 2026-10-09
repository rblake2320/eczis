# UNFILED-IP
"""Apply on the Spark inside MASt3R-SLAM/: adds per-keyframe evidence export (ECZIS-only addition)."""
from pathlib import Path

EXPORT = '''

def save_evidence(savedir, filename, timestamps, keyframes: SharedKeyframes):
    """ECZIS export: per-keyframe canonical points, confidence, image, pose, frame id.

    Everything the grader needs to reproject each keyframe into the others; not part of
    upstream MASt3R-SLAM. Poses are Sim3 data (x y z qx qy qz qw [scale]).
    """
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    out = {}
    for i in range(len(keyframes)):
        kf = keyframes[i]
        h, w = [int(v) for v in kf.img_shape.flatten()[:2]]
        out[f"kf{i}_frame_id"] = np.array(int(kf.frame_id))
        out[f"kf{i}_t"] = np.array(float(timestamps[kf.frame_id]))
        out[f"kf{i}_T_WC"] = kf.T_WC.data.detach().cpu().numpy().reshape(-1)
        out[f"kf{i}_X_canon"] = kf.X_canon.detach().cpu().numpy().reshape(h, w, 3).astype(np.float32)
        out[f"kf{i}_conf"] = kf.get_average_conf().detach().cpu().numpy().reshape(h, w).astype(np.float32)
        out[f"kf{i}_img"] = (kf.uimg.detach().cpu().numpy() * 255).astype(np.uint8).reshape(h, w, 3)
        if kf.K is not None:
            out[f"kf{i}_K"] = kf.K.detach().cpu().numpy().astype(np.float32)
    out["n_keyframes"] = np.array(len(keyframes))
    np.savez_compressed(savedir / filename, **out)
'''

ev = Path("mast3r_slam/evaluate.py")
s = ev.read_text()
if "def save_evidence" not in s:
    ev.write_text(s + EXPORT)
    print("evaluate.py: save_evidence added")
else:
    print("evaluate.py: already patched")

m = Path("main.py")
s = m.read_text()
anchor = '''        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )'''
call = '''
        eval.save_evidence(save_dir, f"{seq_name}_evidence.npz", dataset.timestamps, keyframes)'''
if "save_evidence" not in s:
    assert anchor in s, "save_keyframes call site not found"
    m.write_text(s.replace(anchor, anchor + call, 1))
    print("main.py: save_evidence call added")
else:
    print("main.py: already patched")
