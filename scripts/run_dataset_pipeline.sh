#!/bin/bash
# scripts/run_dataset_pipeline.sh
# Overnight dataset pipeline on RunPod.
#
# Usage (in tmux):
#   tmux new -s dataset
#   bash scripts/run_dataset_pipeline.sh 2>&1 | tee /tmp/dataset_pipeline.log

set -e

export PYTHONPATH=/workspace/MeshLex-Research
export HF_TOKEN="${HF_TOKEN}"

HF_REPO="Pthahnix/MeshLex-Patches"
WORK_BASE="/tmp/meshlex"

echo "=========================================="
echo "MeshLex Daft Dataset Pipeline — $(date)"
echo "=========================================="

# Pre-flight resource check
echo ""
echo "[Pre-flight] Resource check..."
df -h /
free -h
echo ""

echo "[Phase 1/4] Objaverse-LVIS streaming..."
python scripts/stream_objaverse_daft.py \
    --hf_repo "$HF_REPO" \
    --batch_size 500 --download_processes 8 \
    --work_dir "${WORK_BASE}/objaverse" --target_faces 1000

echo ""
echo "[Phase 2/4] ShapeNetCore v2 streaming..."
python scripts/stream_shapenet_daft.py \
    --hf_repo "$HF_REPO" \
    --work_dir "${WORK_BASE}/shapenet" --target_faces 1000

echo ""
echo "[Phase 3/4] Generating splits..."
python scripts/generate_splits_daft.py \
    --hf_repo "$HF_REPO" \
    --holdout_count 100 --test_ratio 0.2 --seed 42 \
    --work_dir "${WORK_BASE}/splits"

echo ""
echo "[Phase 4/4] Validating dataset..."
python scripts/validate_dataset_daft.py \
    --hf_repo "$HF_REPO" --work_dir "${WORK_BASE}/validate"

echo ""
echo "=========================================="
echo "Pipeline complete — $(date)"
echo "=========================================="
