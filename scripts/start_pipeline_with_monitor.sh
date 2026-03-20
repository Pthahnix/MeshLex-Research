#!/bin/bash
# Start both monitor daemon and pipeline in tmux

cd /workspace/MeshLex-Research
export PYTHONPATH=/workspace/MeshLex-Research
# HF_TOKEN must be set externally before running this script

# Start monitor daemon in background
nohup /workspace/MeshLex-Research/scripts/monitor_daemon.sh > /tmp/monitor_daemon.log 2>&1 &

# Run pipeline
python scripts/stream_objaverse_daft.py \
    --hf_repo Pthahnix/MeshLex-Patches \
    --batch_size 500 \
    --download_processes 8 \
    --work_dir /tmp/meshlex/objaverse \
    --target_faces 1000
