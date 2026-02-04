#!/bin/bash

# Script to run FLOWER training on LIBERO datasets with 2x A6000 GPUs
# Works both inside Docker and via SLURM submission

set -e

echo "=========================================="
echo "Running FLOWER LIBERO Training on Cluster"
echo "=========================================="

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse benchmark from arguments (default libero_goal)
BENCHMARK="libero_goal"
for arg in "$@"; do
    if [[ $arg == benchmark=* ]]; then
        BENCHMARK="${arg#*=}"
    fi
done

# Validate benchmark name
VALID_BENCHMARKS=("libero_goal" "libero_spatial" "libero_object" "libero_10" "libero_90")
if [[ ! " ${VALID_BENCHMARKS[@]} " =~ " ${BENCHMARK} " ]]; then
    echo "ERROR: Invalid benchmark '${BENCHMARK}'"
    echo ""
    echo "Available benchmarks:"
    echo "  benchmark=libero_goal     - Goal-conditioned tasks (default)"
    echo "  benchmark=libero_spatial  - Spatial reasoning tasks"
    echo "  benchmark=libero_object   - Object manipulation tasks"
    echo "  benchmark=libero_10       - 10 diverse tasks"
    echo "  benchmark=libero_90       - 90 diverse tasks"
    echo ""
    echo "To download:"
    echo "  ./scripts/download_libero_data.sh ${BENCHMARK}"
    echo ""
    exit 1
fi

# Check if dataset exists
LIBERO_DATASET_DIR="${SCRIPT_DIR}/LIBERO/LIBERO/${BENCHMARK}"
if [ ! -d "${LIBERO_DATASET_DIR}" ]; then
    echo "ERROR: LIBERO benchmark '${BENCHMARK}' not found!"
    echo "Expected location: ${LIBERO_DATASET_DIR}"
    echo ""
    echo "To download:"
    echo "  ./scripts/download_libero_data.sh ${BENCHMARK}"
    echo ""
    exit 1
fi

# Verify GPUs
echo ""
echo "Detected GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Create timestamp for unique run identification
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse max_epochs from arguments (default 50 for LIBERO)
MAX_EPOCHS=50
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

# Parse num_workers from arguments (default 8)
NUM_WORKERS=8
for arg in "$@"; do
    if [[ $arg == num_workers=* ]]; then
        NUM_WORKERS="${arg#*=}"
    fi
done

# Parse checkpoint path from arguments (optional - for fine-tuning)
CHECKPOINT=""
for arg in "$@"; do
    if [[ $arg == checkpoint=* ]]; then
        CHECKPOINT="${arg#*=}"
    fi
done

echo "Configuration:"
echo "  Benchmark: ${BENCHMARK}"
echo "  Dataset path: ${LIBERO_DATASET_DIR}"
echo "  GPUs: 2"
echo "  Batch size: ${BATCH_SIZE} (effective batch: ${BATCH_SIZE}*2*4=32 with grad accum)"
echo "  Max epochs: ${MAX_EPOCHS}"
echo "  Num workers: ${NUM_WORKERS}"
echo "  Multi-GPU: FSDP strategy (memory optimized)"
echo "  EMA: Delayed start (step 5000)"
echo "  Checkpoints: Saved to logs/libero_training_${BENCHMARK}_${TIMESTAMP}/checkpoints"
if [ -n "${CHECKPOINT}" ]; then
    echo "  Fine-tuning from: ${CHECKPOINT}"
fi
echo ""

# Set CUDA memory allocator config for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Build checkpoint argument if provided
CKPT_ARG=""
if [ -n "${CHECKPOINT}" ]; then
    CKPT_ARG="checkpoint=${CHECKPOINT}"
fi

# Run training with memory-optimized settings
# FSDP strategy shards model/optimizer across GPUs to reduce per-GPU memory
# EMA delayed to avoid 4GB extra memory during early training
# Checkpoints auto-saved by Lightning: last.ckpt + best based on val loss
python ${SCRIPT_DIR}/flower/training.py \
    datamodule=libero \
    datamodule.datasets=[${BENCHMARK}] \
    datamodule.root=${SCRIPT_DIR}/LIBERO \
    datamodule.batch_size=${BATCH_SIZE} \
    datamodule.num_workers=${NUM_WORKERS} \
    log_dir=${SCRIPT_DIR}/logs/libero_training_${BENCHMARK}_${TIMESTAMP} \
    max_epochs=${MAX_EPOCHS} \
    devices=2 \
    logger.entity=VLA-Thesis \
    logger.project=libero_a6000 \
    logger.group=${BENCHMARK} \
    logger.name=${BENCHMARK}_2gpu_${TIMESTAMP} \
    model.freeze_florence=False \
    model.freeze_vision_tower=False \
    callbacks.ema.start_step=5000 \
    callbacks.checkpoint.save_top_k=3 \
    callbacks.checkpoint.monitor=val/loss \
    callbacks.checkpoint.mode=min \
    callbacks.checkpoint.save_last=True \
    ${CKPT_ARG}

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Logs: ${SCRIPT_DIR}/logs/libero_training_${BENCHMARK}_${TIMESTAMP}"
echo "Checkpoints: ${SCRIPT_DIR}/logs/libero_training_${BENCHMARK}_${TIMESTAMP}/checkpoints/"
echo "  - last.ckpt (most recent)"
echo "  - epoch=X-step=Y.ckpt (top 3 best by val/loss)"
echo ""
echo "To backup checkpoints:"
echo "  mkdir -p ~/backups/libero_${BENCHMARK}_${TIMESTAMP}"
echo "  cp -r ${SCRIPT_DIR}/logs/libero_training_${BENCHMARK}_${TIMESTAMP}/checkpoints ~/backups/libero_${BENCHMARK}_${TIMESTAMP}/"
echo ""
