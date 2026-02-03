#!/bin/bash

# Script to run FLOWER evaluation on CALVIN debug dataset

set -e

echo "=========================================="
echo "Running FLOWER Evaluation on CALVIN"
echo "=========================================="

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if checkpoint exists
if [ ! -f "${SCRIPT_DIR}/checkpoints/calvin_d/model.safetensors" ]; then
    echo "ERROR: Calvin D checkpoint not found!"
    echo "Please run: scripts/download_calvin_checkpoint.sh D first"
    exit 1
fi

# Check if dataset exists
if [ ! -d "${SCRIPT_DIR}/dataset/calvin_debug_dataset" ]; then
    echo "ERROR: CALVIN debug dataset not found!"
    echo "Please run: scripts/download_data.sh debug"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Checkpoint: ${SCRIPT_DIR}/checkpoints/calvin_d/model.safetensors"
echo "  Dataset: ${SCRIPT_DIR}/dataset/calvin_debug_dataset"
echo "  Device: cuda:0"
echo ""

# Run evaluation with absolute paths (Hydra changes CWD, so relative paths won't work)
python flower/evaluation/flower_evaluate.py \
    train_folder=${SCRIPT_DIR}/checkpoints/calvin_d \
    checkpoint=${SCRIPT_DIR}/checkpoints/calvin_d/model.safetensors \
    dataset_path=${SCRIPT_DIR}/dataset/calvin_debug_dataset \
    log_dir=${SCRIPT_DIR}/evaluation/calvin_debug_evaluation \
    wandb_entity=VLA-Thesis\
    device=0 \
    num_sequences=100 \
    num_videos=5 \
    log_wandb=true

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
