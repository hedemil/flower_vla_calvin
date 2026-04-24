#!/bin/bash

# Script to run FLOWER evaluation on LIBERO benchmarks
#
# Usage:
#   ./scripts/run_libero_evaluation.sh <benchmark> <model>
#
# Examples:
#   ./scripts/run_libero_evaluation.sh libero_spatial meanflower
#   ./scripts/run_libero_evaluation.sh libero_spatial flower
#   ./scripts/run_libero_evaluation.sh libero_goal meanflower

set -e

echo "=========================================="
echo "Running FLOWER Evaluation on LIBERO"
echo "=========================================="

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==========================================="
echo "Script Directory: $SCRIPT_DIR"
echo "==========================================="

# Parse arguments
BENCHMARK="${1:-libero_spatial}"
MODEL="${2:-meanflower}"
echo "Selected Benchmark: $BENCHMARK"
echo "Selected Model:     $MODEL"

# Validate benchmark
VALID_BENCHMARKS=("libero_spatial" "libero_object" "libero_goal" "libero_10" "libero_90")
if [[ ! " ${VALID_BENCHMARKS[@]} " =~ " ${BENCHMARK} " ]]; then
    echo "ERROR: Invalid benchmark: $BENCHMARK"
    echo "Valid options: ${VALID_BENCHMARKS[@]}"
    exit 1
fi

# Validate model
VALID_MODELS=("flower" "meanflower")
if [[ ! " ${VALID_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "ERROR: Invalid model: $MODEL"
    echo "Valid options: ${VALID_MODELS[@]}"
    exit 1
fi

# Set num_sampling_steps based on model
if [[ "$MODEL" == "meanflower" ]]; then
    NUM_SAMPLING_STEPS=1
else
    NUM_SAMPLING_STEPS=4
fi

# Checkpoint directory: checkpoints/<model>/<benchmark>/
CHECKPOINT_DIR="/workspace/flower_vla_calvin/checkpoints/${MODEL}/${BENCHMARK}"
echo "Checkpoint Directory: $CHECKPOINT_DIR"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ] || [ ! "$(ls -A $CHECKPOINT_DIR)" ]; then
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_DIR"
    echo ""
    echo "Expected structure:"
    echo "  checkpoints/${MODEL}/${BENCHMARK}/"
    echo "    ├── .hydra/config.yaml"
    echo "    └── model.ckpt (or model.safetensors)"
    exit 1
fi

# Auto-detect checkpoint file
if [ -f "$CHECKPOINT_DIR/model.ckpt" ]; then
    CKPT_FILE="$CHECKPOINT_DIR/model.ckpt"
elif [ -f "$CHECKPOINT_DIR/model.safetensors" ]; then
    CKPT_FILE="$CHECKPOINT_DIR/model.safetensors"
else
    # Find any .ckpt file
    CKPT_FILE=$(find "$CHECKPOINT_DIR" -name "*.ckpt" -type f | head -1)
    if [ -z "$CKPT_FILE" ]; then
        echo "ERROR: No .ckpt or .safetensors file found in $CHECKPOINT_DIR"
        exit 1
    fi
fi

# Auto-detect hydra config
TRAIN_FOLDER="$CHECKPOINT_DIR/.hydra/config.yaml"
if [ ! -f "$TRAIN_FOLDER" ]; then
    # Search parent directories
    TRAIN_FOLDER=$(find "$CHECKPOINT_DIR" -path "*/.hydra/config.yaml" -type f | head -1)
    if [ -z "$TRAIN_FOLDER" ]; then
        echo "ERROR: .hydra/config.yaml not found in $CHECKPOINT_DIR"
        exit 1
    fi
fi

# Check if LIBERO dataset exists
LIBERO_DATASET_DIR="/workspace/flower_vla_calvin/LIBERO/libero/datasets/${BENCHMARK}"
if [ ! -d "$LIBERO_DATASET_DIR" ] || [ ! "$(ls -A $LIBERO_DATASET_DIR)" ]; then
    echo "WARNING: LIBERO dataset not found at: $LIBERO_DATASET_DIR"
    echo "  LIBERO benchmark API may still work if ~/.libero/config.yaml is set."
fi

echo ""
echo "Configuration:"
echo "  Benchmark:           $BENCHMARK"
echo "  Model:               $MODEL"
echo "  Checkpoint:          $CKPT_FILE"
echo "  Train folder:        $TRAIN_FOLDER"
echo "  Num sampling steps:  $NUM_SAMPLING_STEPS"
echo "  Device:              cuda:0"
echo ""

# Run evaluation with Hydra
python flower/evaluation/flower_eval_libero.py \
    train_folder="$TRAIN_FOLDER" \
    checkpoint="$CKPT_FILE" \
    benchmark_name="$BENCHMARK" \
    num_sampling_steps="$NUM_SAMPLING_STEPS" \
    eval_cfg_overwrite.model.num_sampling_steps="$NUM_SAMPLING_STEPS" \
    log_dir="${SCRIPT_DIR}/evaluation/${BENCHMARK}_${MODEL}_evaluation" \
    wandb_entity=VLA-Thesis \
    device=0 \
    n_eval=5 \
    max_steps=520 \
    num_videos=5 \
    log_wandb=true

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved in evaluation/${BENCHMARK}_${MODEL}_evaluation/"
