# UNFILED-IP
"""Send a recorded sweep to the DGX Spark, run MASt3R-SLAM there, pull the result back, grade it.

    uv run python tools/spark_scan.py artifacts/earscan/<time>            # uses frames.npz in that folder
    uv run python tools/spark_scan.py artifacts/earscan/<time> --lens lens.json --host spark1

Transfer is scp over the private link (ssh config alias; jump host handled there). Nothing
leaves the two machines. The Spark side must have been prepared with tools/spark_mast3r_setup.sh.
Output: <folder>/mast3r/{<name>.ply,<name>.txt,graded.ply,result.json}. EXPERIMENTAL.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REMOTE_ROOT = "~/eczis-3d"
SLAM_DIR = REMOTE_ROOT + "/MASt3R-SLAM"


def sh(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def ssh(host: str, remote_cmd: str, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    return sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, remote_cmd], timeout)


def run(folder: Path, host: str, lens: Path | None, stride: int) -> dict[str, object]:
    t0 = time.perf_counter()
    frames_file = folder / "frames.npz"
    if not frames_file.exists():
        raise SystemExit(f"no frames.npz in {folder}")
    name = folder.name.replace("-", "")
    out = folder / "mast3r"
    out.mkdir(exist_ok=True)
    frames = np.load(frames_file)["frames"][::stride]
    h, w = frames.shape[1:3]
    # 1. push frames as PNGs (one archive)
    from PIL import Image

    png_dir = folder / "_png"
    png_dir.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).convert("RGB").save(png_dir / f"{i:04d}.png")
    import shutil

    archive = shutil.make_archive(str(folder / name), "zip", png_dir)
    shutil.rmtree(png_dir)
    r = sh(["scp", "-o", "BatchMode=yes", "-q", archive, f"{host}:{REMOTE_ROOT}/{name}.zip"], 600)
    if r.returncode:
        raise SystemExit(f"scp push failed: {r.stderr.strip()}")
    Path(archive).unlink()
    t_push = time.perf_counter() - t0
    # 2. run SLAM remotely (calibration-free unless a lens file is given)
    calib = ""
    if lens is not None:
        from eczis.lab.calibrate import LensProfile

        lp = LensProfile.from_json(lens)
        intr = f"{SLAM_DIR}/config/intr_{name}.yaml"
        yaml = f"width: {lp.width}\nheight: {lp.height}\ncalibration: [{lp.fx}, {lp.fy}, {lp.cx}, {lp.cy}]\n"
        ssh(host, f"cat > {intr} <<'EOF'\n{yaml}EOF")
        calib = f" --calib {intr}"
    remote = (f"cd {REMOTE_ROOT} && rm -rf {name} && mkdir {name} && unzip -q {name}.zip -d {name} && rm {name}.zip && "
              f"cd {SLAM_DIR} && . {REMOTE_ROOT}/.venv/bin/activate && export PATH=/usr/local/cuda-13.0/bin:$PATH && "
              f"rm -rf logs/{name} && python main.py --dataset {REMOTE_ROOT}/{name} --config config/base.yaml "
              f"--no-viz --save-as {name}{calib} 2>&1 | grep -E 'FPS|Skipped|RELOCAL|Failed|done|Error|error' | tail -12; "
              f"ls logs/{name}/{name}.ply logs/{name}/{name}.txt")
    r = ssh(host, remote)
    log_tail = (r.stdout + r.stderr).strip().splitlines()[-14:]
    if r.returncode or f"logs/{name}/{name}.ply" not in r.stdout:
        raise SystemExit("remote SLAM failed:\n" + "\n".join(log_tail))
    t_slam = time.perf_counter() - t0 - t_push
    # 3. pull results
    for ext in ("ply", "txt"):
        r = sh(["scp", "-o", "BatchMode=yes", "-q", f"{host}:{SLAM_DIR}/logs/{name}/{name}.{ext}", str(out / f"{name}.{ext}")], 600)
        if r.returncode:
            raise SystemExit(f"scp pull failed: {r.stderr.strip()}")
    # 4. grade locally with ECZIS rules
    from eczis.lab.grade_dense import grade_mast3r_output

    if lens is not None:
        k = LensProfile.from_json(lens).matrix()
        lens_note = f"calibrated lens {lens.name}"
    else:
        k = np.array([[0.8 * w, 0, w / 2], [0, 0.8 * w, h / 2], [0, 0, 1.0]])
        lens_note = "UNCALIBRATED lens (0.8 x width guess); NO SCALE"
    g = grade_mast3r_output(out / f"{name}.ply", out / f"{name}.txt", k, (w, h), out / "graded.ply")
    result = {**g, "lens": lens_note, "frames_sent": int(len(frames)), "host": host,
              "seconds": {"push": round(t_push, 1), "slam": round(t_slam, 1), "total": round(time.perf_counter() - t0, 1)},
              "slam_log_tail": log_tail, "measurement_record_not_diagnosis": True}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--host", default="spark1")
    ap.add_argument("--lens", type=Path, default=None, help="LensProfile JSON from the calibrator")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    res = run(a.folder, a.host, a.lens, a.stride)
    print(json.dumps({k: res[k] for k in ("points", "keyframes", "counts", "fractions", "lens", "seconds")}, indent=2))
    print("graded cloud:", res["graded_ply"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
