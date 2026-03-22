#!/bin/bash
# Post-training automation for full-scale VQ-VAE training
# Run after training completes to upload checkpoints and start encoding
#
# Usage: bash scripts/post_training_automation.sh <job_name>
# job_name: pca | nopca | pca_k512 | pca_k2048 | all
#
# This script:
# 1. Uploads checkpoint to HuggingFace
# 2. Starts token encoding on the freed GPU

set -e

source ~/.bashrc
export PYTHONPATH=/home/pthahnix/MeshLex-Research
cd /home/pthahnix/MeshLex-Research

DATA_BASE=/data/pthahnix/MeshLex-Research
ARROW_BASE=$DATA_BASE/datasets/MeshLex-Patches/splits
FEAT_BASE=$DATA_BASE/datasets/MeshLex-Patches/features
CKPT_BASE=$DATA_BASE/checkpoints
SEQ_BASE=$DATA_BASE/sequences
LOG_DIR=results/fullscale_eval

mkdir -p $LOG_DIR

JOB=${1:-"all"}

upload_checkpoint() {
    local exp_name=$1
    local ckpt_dir=$2
    echo "=== Uploading $exp_name checkpoint to HuggingFace ==="
    python3 - <<EOF
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get("HF_TOKEN"))

exp_name = "$exp_name"
ckpt_dir = "$ckpt_dir"

import os
for fname in ["checkpoint_final.pt", "training_history.json", "config.json"]:
    fpath = os.path.join(ckpt_dir, fname)
    if os.path.exists(fpath):
        print(f"Uploading {fname}...")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"checkpoints/{exp_name}/{fname}",
            repo_id="Pthahnix/MeshLex-Research",
            repo_type="model",
        )
        print(f"  ✅ Uploaded {fname}")
    else:
        print(f"  ⚠ {fname} not found, skipping")

print(f"✅ Checkpoint uploaded to HF: Pthahnix/MeshLex-Research/checkpoints/{exp_name}/")
EOF
}

encode_sequences() {
    local config=$1   # pca | nopca | pca_k512 | pca_k2048
    local gpu=$2
    local ckpt=$3
    local feat_dir=$4
    local out_dir=$5
    local extra_args=${6:-""}

    echo "=== Encoding sequences: $config on GPU $gpu ==="
    mkdir -p $out_dir
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. python3 scripts/encode_sequences.py \
        --arrow_dir $ARROW_BASE/seen_train \
        --feature_dir $feat_dir \
        --checkpoint $ckpt \
        --output_dir $out_dir \
        --mode rvq --batch_size 4096 \
        $extra_args \
        2>&1 | tee $LOG_DIR/encode_${config}.log

    count=$(ls $out_dir/*_sequence.npz 2>/dev/null | wc -l)
    echo "✅ Encoded $count sequences for $config → $out_dir"
}

run_pca_k512() {
    echo "=== POST-TRAINING: PCA K=512 ==="
    upload_checkpoint "rvq_full_pca_k512" "$CKPT_BASE/rvq_full_pca_k512"
    encode_sequences "pca_k512" "1" \
        "$CKPT_BASE/rvq_full_pca_k512/checkpoint_final.pt" \
        "$FEAT_BASE/seen_train" \
        "$SEQ_BASE/rvq_full_pca_k512"
}

run_pca_k2048() {
    echo "=== POST-TRAINING: PCA K=2048 ==="
    upload_checkpoint "rvq_full_pca_k2048" "$CKPT_BASE/rvq_full_pca_k2048"
    encode_sequences "pca_k2048" "1" \
        "$CKPT_BASE/rvq_full_pca_k2048/checkpoint_final.pt" \
        "$FEAT_BASE/seen_train" \
        "$SEQ_BASE/rvq_full_pca_k2048"
}

run_pca() {
    echo "=== POST-TRAINING: PCA K=1024 ==="
    upload_checkpoint "rvq_full_pca" "$CKPT_BASE/rvq_full_pca"
    encode_sequences "pca" "0" \
        "$CKPT_BASE/rvq_full_pca/checkpoint_final.pt" \
        "$FEAT_BASE/seen_train" \
        "$SEQ_BASE/rvq_full_pca"
}

run_nopca() {
    echo "=== POST-TRAINING: noPCA K=1024 ==="
    upload_checkpoint "rvq_full_nopca" "$CKPT_BASE/rvq_full_nopca"
    encode_sequences "nopca" "0" \
        "$CKPT_BASE/rvq_full_nopca/checkpoint_final.pt" \
        "$FEAT_BASE/seen_train_nopca" \
        "$SEQ_BASE/rvq_full_nopca" \
        "--nopca"
}

case $JOB in
    pca_k512)  run_pca_k512 ;;
    pca_k2048) run_pca_k2048 ;;
    pca)       run_pca ;;
    nopca)     run_nopca ;;
    all)
        # GPU 2 jobs finish first
        if [ -f "$CKPT_BASE/rvq_full_pca_k512/checkpoint_final.pt" ] && \
           [ ! -f "$SEQ_BASE/rvq_full_pca_k512/.done" ]; then
            run_pca_k512
            touch "$SEQ_BASE/rvq_full_pca_k512/.done"
        fi
        if [ -f "$CKPT_BASE/rvq_full_pca_k2048/checkpoint_final.pt" ] && \
           [ ! -f "$SEQ_BASE/rvq_full_pca_k2048/.done" ]; then
            run_pca_k2048
            touch "$SEQ_BASE/rvq_full_pca_k2048/.done"
        fi
        # GPU 1 jobs finish later
        if [ -f "$CKPT_BASE/rvq_full_pca/checkpoint_final.pt" ] && \
           [ ! -f "$SEQ_BASE/rvq_full_pca/.done" ]; then
            run_pca
            touch "$SEQ_BASE/rvq_full_pca/.done"
        fi
        if [ -f "$CKPT_BASE/rvq_full_nopca/checkpoint_final.pt" ] && \
           [ ! -f "$SEQ_BASE/rvq_full_nopca/.done" ]; then
            run_nopca
            touch "$SEQ_BASE/rvq_full_nopca/.done"
        fi
        ;;
    *)
        echo "Unknown job: $JOB. Use: pca | nopca | pca_k512 | pca_k2048 | all"
        exit 1
        ;;
esac

echo "=== post_training_automation.sh DONE for $JOB ==="
