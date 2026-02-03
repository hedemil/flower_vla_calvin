#!/bin/bash

# Script to run FLOWER training on cluster with 2x A6000 GPUs
# Works both inside Docker and via SLURM submission

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
echo ""
echo "Detected GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Create timestamp for unique run identification
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse max_epochs from arguments (default 100)
MAX_EPOCHS=10
for arg in "$@"; do
    if [[ $arg == max_epochs=* ]]; then
        MAX_EPOCHS="${arg#*=}"
    fi
done

# Parse batch_size from arguments (default 4)
BATCH_SIZE=4
for arg in "$@"; do
    if [[ $arg == batch_size=* ]]; then
        BATCH_SIZE="${arg#*=}"
    fi
done

echo "Configuration:"
echo "  Dataset: ${SCRIPT_DIR}/dataset/calvin_debug_dataset"
echo "  GPUs: 2"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max epochs: ${MAX_EPOCHS}"
echo "  Multi-GPU: FSDP strategy (memory optimized)"
echo "  Camera views: Single (static only)"
echo "  EMA: Delayed start (step 5000)"
echo ""

# Set CUDA memory allocator config for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training with memory-optimized settings
# FSDP strategy shards model/optimizer across GPUs to reduce per-GPU memory
# EMA delayed to avoid 4GB extra memory during early training
python ${SCRIPT_DIR}/flower/training_calvin.py \
    log_dir=${SCRIPT_DIR}/logs/training_${TIMESTAMP} \
    batch_size=${BATCH_SIZE} \
    max_epochs=${MAX_EPOCHS} \
    devices=2 \
    logger.entity=VLA-Thesis \
    logger.project=calvin_a6000 \
    logger.group=cluster_training \
    logger.name=a6000_2gpu_fsdp_${TIMESTAMP} \
    model.freeze_florence=False \
    model.freeze_vision_tower=False \
    model.use_second_view=False \
    trainer.limit_train_batches=10 \
    rollout_lh_skip_epochs=100 \
    callbacks.ema.start_step=100000


echo ""
echo "Training completed!"
echo "Logs saved to: ${SCRIPT_DIR}/logs/training_${TIMESTAMP}"
