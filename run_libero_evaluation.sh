#!/bin/bash

# Script to run FLOWER evaluation on LIBERO benchmarks

set -e

echo "=========================================="
echo "Running FLOWER Evaluation on LIBERO"
echo "=========================================="

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
BENCHMARK="${1:-libero_spatial}"
CHECKPOINT_DIR="${2:-${SCRIPT_DIR}/checkpoints/${BENCHMARK}}"

# Validate benchmark
VALID_BENCHMARKS=("libero_spatial" "libero_object" "libero_goal" "libero_10" "libero_90")
if [[ ! " ${VALID_BENCHMARKS[@]} " =~ " ${BENCHMARK} " ]]; then
    echo "ERROR: Invalid benchmark: $BENCHMARK"
    echo "Valid options: ${VALID_BENCHMARKS[@]}"
    exit 1
fi

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ] || [ ! "$(ls -A $CHECKPOINT_DIR)" ]; then
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_DIR"
    echo ""
    echo "To download checkpoint, run:"
    echo "  ./download_libero_checkpoint.sh $BENCHMARK"
    exit 1
fi

# Check if LIBERO dataset exists
LIBERO_DATASET_DIR="${SCRIPT_DIR}/LIBERO/libero/datasets/${BENCHMARK}"
if [ ! -d "$LIBERO_DATASET_DIR" ] || [ ! "$(ls -A $LIBERO_DATASET_DIR)" ]; then
    echo "ERROR: LIBERO dataset not found at: $LIBERO_DATASET_DIR"
    echo ""
    echo "To download dataset, run:"
    echo "  cd LIBERO && python benchmark_scripts/download_libero_datasets.py --datasets $BENCHMARK --use-huggingface"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Benchmark: $BENCHMARK"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Dataset: $LIBERO_DATASET_DIR"
echo "  Device: cuda:0"
echo ""

# Convert benchmark name to uppercase for Hydra config
# BENCHMARK_UPPER=$(echo "$BENCHMARK" | tr '[:lower:]' '[:upper:]')

# Run evaluation with Hydra
python flower/evaluation/flower_eval_libero.py \
    train_folder="$CHECKPOINT_DIR" \
    checkpoint="$CHECKPOINT_DIR/model.safetensors" \
    benchmark_name="$BENCHMARK" \
    log_dir="${SCRIPT_DIR}/evaluation/${BENCHMARK}_evaluation" \
    wandb_entity=VLA-Thesis \
    device=0 \
    n_eval=20 \
    max_steps=520 \
    num_videos=5 \
    log_wandb=true

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved in outputs/ directory"
echo "Videos saved in outputs/videos/ directory (if enabled)"
echo ""
