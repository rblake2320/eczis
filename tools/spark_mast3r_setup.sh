#!/usr/bin/env bash
# UNFILED-IP
# Reproducible MASt3R-SLAM setup on DGX Spark (GB10, aarch64, CUDA 13.0, torch 2.14 cu130).
# Verified 2026-10-07 on spark-3cdf. Run on the Spark: bash spark_mast3r_setup.sh
set -euo pipefail
ROOT=~/eczis-3d; mkdir -p "$ROOT"; cd "$ROOT"
# 1. venv on a Python that ships its C headers (system python3-dev was unavailable: stale apt index)
[ -d .venv ] || ~/miniforge3/bin/python -m venv .venv
. .venv/bin/activate
pip install -q --index-url https://download.pytorch.org/whl/cu130 torch torchvision
pip install -q ninja setuptools wheel numpy pillow
[ -d MASt3R-SLAM ] || git clone -q --recursive https://github.com/rmurai0610/MASt3R-SLAM.git
cd MASt3R-SLAM
export PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST="12.1" MAX_JOBS=10
# 2. CUDA 13 dropped the old -gencode targets hard-coded in setup.py -> GB10 only
python - <<'PY'
import re; p="setup.py"; s=open(p).read().splitlines(); out=[]; done=False
for ln in s:
    if "-gencode=arch=compute_" in ln:
        if not done: out.append(ln.split('"')[0]+'"-gencode=arch=compute_121,code=sm_121",'); done=True
        continue
    out.append(ln)
open(p,"w").write("\n".join(out)+"\n")
PY
# 3. torch >= 2.x API renames in the CUDA sources
grep -rl "AT_DISPATCH.*\.type()" --include=*.cu --include=*.cpp . | xargs -r sed -i -E 's/(AT_DISPATCH[A-Z_0-9]*\()([A-Za-z_0-9]+)\.type\(\)/\1\2.scalar_type()/g'
sed -i 's/torch::linalg::linalg_norm/torch::linalg_norm/g' mast3r_slam/backend/src/gn_kernels.cu
# 4. torch.load default changed (weights_only=True) -> allow the trusted checkpoints
grep -rl "torch.load(" --include=*.py mast3r_slam thirdparty/mast3r/mast3r thirdparty/mast3r/dust3r/dust3r \
  | xargs -r sed -i -E 's/torch\.load\(([^)]*)\)/torch.load(\1, weights_only=False)/g; s/weights_only=False, weights_only=False/weights_only=False/g'
# 5. build without pip isolation so torch is visible to the extension builds (in3d viewer skipped: headless)
pip install -q --no-build-isolation -e thirdparty/mast3r
echo "$ROOT/MASt3R-SLAM/thirdparty/mast3r/dust3r" > "$(python -c 'import site;print(site.getsitepackages()[0])')/dust3r_path.pth"
pip install -q --no-build-isolation -e .
# 5b. ECZIS per-keyframe evidence export (X_canon, conf, image, pose per keyframe)
python3 "$(dirname "$0")/spark_mast3r_evidence_patch.py" 2>/dev/null || python3 ~/eczis-3d/spark_mast3r_evidence_patch.py
# 6. checkpoints
mkdir -p checkpoints && cd checkpoints
for f in MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl; do
  [ -s "$f" ] || wget -q -nv "https://download.europe.naverlabs.com/ComputerVision/MASt3R/$f"
done
cd .. && python -c "import torch, curope, mast3r, dust3r, lietorch, mast3r_slam; print('MASt3R-SLAM ready on', torch.cuda.get_device_name(0))"
# Run: python main.py --dataset <image folder> --config config/base.yaml --no-viz --save-as <name>
