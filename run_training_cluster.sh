#!/bin/bash

# Script to run FLOWER training on cluster with 2x A6000 GPUs

set -e

echo "=========================================="
echo "Running FLOWER Training on Cluster"
echo "=========================================="

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if datasets exist
if [ ! -d "${SCRIPT_DIR}/dataset/calvin_debug_dataset" ]; then
    echo "ERROR: CALVIN debug dataset not found!"
    echo "Please run: ./download_data.sh debug"
    exit 1
fi

# Verify GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo ""
echo "Detected GPUs: ${GPU_COUNT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Expected 2 GPUs but found ${GPU_COUNT}"
    echo "Continuing anyway..."
fi

echo ""
echo "Configuration:"
echo "  Dataset: ${SCRIPT_DIR}/dataset/calvin_debug_dataset"
echo "  GPUs: ${GPU_COUNT}"
echo "  Batch size: 64"
echo "  Max epochs: 100"
echo "  Multi-GPU: DDP strategy"
echo ""

# Create timestamp for unique run identification
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run training with cluster-optimized settings
python flower/training_calvin.py \
    root_data_dir=${SCRIPT_DIR}/dataset/calvin_debug_dataset \
    lang_folder=lang_annotations \
    log_dir=${SCRIPT_DIR}/logs/training_${TIMESTAMP} \
    batch_size=64 \
    max_epochs=100 \
    devices=2 \
    logger.entity=VLA-Thesis \
    logger.project=calvin_a6000 \
    logger.group=cluster_training \
    logger.name=a6000_2gpu_${TIMESTAMP} \
    model.freeze_florence=False \
    model.freeze_vision_tower=False \
    model.use_second_view=True \
    device=0
