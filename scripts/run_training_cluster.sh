#!/bin/bash

# Script to run FLOWER training on cluster with 2x A6000 GPUs
# Works both inside Docker and via SLURM submission

set -e

echo "=========================================="
echo "Running FLOWER Training on Cluster"
echo "=========================================="

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse dataset from arguments (default debug for testing, use 'D' for full training)
DATASET="calvin_debug_dataset"
for arg in "$@"; do
    if [[ $arg == dataset=* ]]; then
        DATASET_ARG="${arg#*=}"
        case $DATASET_ARG in
            debug)
                DATASET="calvin_debug_dataset"
                ;;
            D|d|calvin_d)
                DATASET="task_D_D"
                ;;
            ABC|abc)
                DATASET="task_ABC_D"
                ;;
            ABCD|abcd)
                DATASET="task_ABCD_D"
                ;;
            *)
                DATASET="$DATASET_ARG"  # Allow custom dataset names
                ;;
        esac
    fi
done

# Check if dataset exists
if [ ! -d "${SCRIPT_DIR}/dataset/${DATASET}" ]; then
    echo "ERROR: Dataset '${DATASET}' not found!"
    echo ""
    echo "Available options:"
    echo "  dataset=debug  -> calvin_debug_dataset (quick testing)"
    echo "  dataset=D      -> task_D_D (single env, ~50GB)"
    echo "  dataset=ABC    -> task_ABC_D (3 envs, ~150GB)"
    echo "  dataset=ABCD   -> task_ABCD_D (4 envs, ~200GB)"
    echo ""
    echo "To download:"
    echo "  ./download_data.sh debug     # Debug dataset"
    echo "  ./download_data.sh D         # Calvin D dataset"
    echo "  ./download_data.sh ABC       # Calvin ABC dataset"
    echo "  ./download_data.sh ABCD      # Full Calvin dataset"
    echo ""
    echo "After download, preprocess with:"
    echo "  ./docker_run.sh"
    echo "  python preprocess/extract_by_key.py -i /workspace/flower_vla_calvin/dataset --in_task ${DATASET} --in_split all -k rel_actions"
    exit 1
fi

# Verify GPUs
echo ""
echo "Detected GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Create timestamp for unique run identification
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse max_epochs from arguments (default 25 for D->D training, ~40k steps)
MAX_EPOCHS=25
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

# Parse rollout_skip from arguments (default 20 - evaluate near end)
ROLLOUT_SKIP=20
for arg in "$@"; do
    if [[ $arg == rollout_skip=* ]]; then
        ROLLOUT_SKIP="${arg#*=}"
    fi
done

echo "Configuration:"
echo "  Dataset: ${DATASET}"
echo "  Dataset path: ${SCRIPT_DIR}/dataset/${DATASET}"
echo "  GPUs: 2"
echo "  Batch size: ${BATCH_SIZE} (effective batch: ${BATCH_SIZE}*2*4=32 with grad accum)"
echo "  Max epochs: ${MAX_EPOCHS}"
echo "  Rollout evaluation after: ${ROLLOUT_SKIP} epochs"
echo "  Multi-GPU: FSDP strategy (memory optimized)"
echo "  Camera views: Single (static only)"
echo "  EMA: Delayed start (step 5000)"
echo "  Checkpoints: Saved to logs/training_${TIMESTAMP}/checkpoints"
echo ""

# Set CUDA memory allocator config for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Determine benchmark name for logging
case $DATASET in
    calvin_debug_dataset)
        BENCHMARK_NAME="calvin_debug"
        ;;
    task_D_D)
        BENCHMARK_NAME="calvin_d"
        ;;
    task_ABC_D)
        BENCHMARK_NAME="calvin_abc"
        ;;
    task_ABCD_D)
        BENCHMARK_NAME="calvin_abcd"
        ;;
    *)
        BENCHMARK_NAME="calvin_custom"
        ;;
esac

# Run training with memory-optimized settings
# FSDP strategy shards model/optimizer across GPUs to reduce per-GPU memory
# EMA delayed to avoid 4GB extra memory during early training
# Checkpoints auto-saved by Lightning: last.ckpt + best based on val loss
python ${SCRIPT_DIR}/flower/training_calvin.py \
    root_data_dir=${SCRIPT_DIR}/dataset/${DATASET} \
    benchmark_name=${BENCHMARK_NAME} \
    log_dir=${SCRIPT_DIR}/logs/training_${TIMESTAMP} \
    batch_size=${BATCH_SIZE} \
    max_epochs=${MAX_EPOCHS} \
    devices=2 \
    logger.entity=VLA-Thesis \
    logger.project=calvin_a6000 \
    logger.group=${BENCHMARK_NAME} \
    logger.name=${BENCHMARK_NAME}_2gpu_${TIMESTAMP} \
    model.freeze_florence=False \
    model.freeze_vision_tower=False \
    model.use_second_view=False \
    rollout_lh_skip_epochs=${ROLLOUT_SKIP} \
    callbacks.ema.start_step=5000 \
    callbacks.checkpoint.save_top_k=3 \
    callbacks.checkpoint.monitor=val/loss \
    callbacks.checkpoint.mode=min \
    callbacks.checkpoint.save_last=True


echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Logs: ${SCRIPT_DIR}/logs/training_${TIMESTAMP}"
echo "Checkpoints: ${SCRIPT_DIR}/logs/training_${TIMESTAMP}/checkpoints/"
echo "  - last.ckpt (most recent)"
echo "  - epoch=X-step=Y.ckpt (top 3 best by val/loss)"
echo ""
echo "To backup checkpoints:"
echo "  mkdir -p ~/backups/calvin_d2d_${TIMESTAMP}"
echo "  cp -r ${SCRIPT_DIR}/logs/training_${TIMESTAMP}/checkpoints ~/backups/calvin_d2d_${TIMESTAMP}/"
echo ""
