#!/bin/bash

# Script to run FLOWER evaluation on LIBERO benchmarks
#
# Usage:
#   ./scripts/run_libero_evaluation.sh <benchmark|all|csv> <model> [nfe|all|csv]
#
# Examples (single benchmark):
#   ./scripts/run_libero_evaluation.sh libero_spatial meanflower
#   ./scripts/run_libero_evaluation.sh libero_spatial flower
#   ./scripts/run_libero_evaluation.sh libero_goal meanflower
#   ./scripts/run_libero_evaluation.sh libero_10 flower_ablation 1
#   ./scripts/run_libero_evaluation.sh libero_10 flower_ablation 2
#   ./scripts/run_libero_evaluation.sh libero_10 flower_ablation 3
#
# Examples (cross-NFE sweep on a single ckpt — isolates inference NFE):
#   ./scripts/run_libero_evaluation.sh libero_10 flower_cross_nfe 1,2,3,4,8
#     # -> evaluates checkpoints/flower/libero_10/ at NFE=1,2,3,4,8
#   ./scripts/run_libero_evaluation.sh libero_10 flower_cross_nfe all
#     # -> default sweep: NFE=1,2,3,4,6,8
#   CROSS_NFE_CKPT_DIR=checkpoints/flower_ablation/libero_10_nfe2 \
#     ./scripts/run_libero_evaluation.sh libero_10 flower_cross_nfe 1,2,3,4,8
#     # -> override source ckpt dir
#
# Examples (multi-benchmark sweep):
#   ./scripts/run_libero_evaluation.sh all flower
#     # -> libero_spatial, libero_object, libero_goal, libero_10
#   ./scripts/run_libero_evaluation.sh libero_spatial,libero_object flower
#   ./scripts/run_libero_evaluation.sh libero_10 flower_ablation all
#     # -> NFE=1, NFE=2, NFE=3 on libero_10
#   ./scripts/run_libero_evaluation.sh libero_10 flower_ablation 1,3
#
# Multi-benchmark / multi-NFE sweeps run sequentially on the same GPU. A failure
# in one (benchmark, nfe) combination does not abort the sweep — failures are
# collected and reported in the summary at the end; the script exits non-zero
# if any combination failed.
#
# Env vars:
#   LOG_WANDB (default: true) — set to false to skip W&B uploads for the whole sweep.
#     Example: LOG_WANDB=false ./scripts/run_libero_evaluation.sh all flower
#   CROSS_NFE_CKPT_DIR — only for MODEL=flower_cross_nfe. Overrides the source
#     checkpoint directory (default: checkpoints/flower/<benchmark>).

echo "=========================================="
echo "Running FLOWER Evaluation on LIBERO"
echo "=========================================="

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==========================================="
echo "Script Directory: $SCRIPT_DIR"
echo "==========================================="

# Parse arguments
BENCHMARK_ARG="${1:-libero_spatial}"
MODEL="${2:-meanflower}"
NFE_ARG="${3:-1}"
LOG_WANDB="${LOG_WANDB:-true}"

VALID_BENCHMARKS=("libero_spatial" "libero_object" "libero_goal" "libero_10" "libero_90")
ALL_BENCHMARKS=("libero_spatial" "libero_object" "libero_goal" "libero_10")
VALID_MODELS=("flower" "meanflower" "imf" "decoupled_meanflower" "flower_ablation" "flower_cross_nfe")
VALID_NFES=("1" "2" "3")
ALL_NFES=("1" "2" "3")
ALL_NFES_CROSS=("1" "2" "3" "4")

# Expand benchmark argument
if [[ "$BENCHMARK_ARG" == "all" ]]; then
    BENCHMARKS=("${ALL_BENCHMARKS[@]}")
else
    IFS=',' read -ra BENCHMARKS <<<"$BENCHMARK_ARG"
fi

# Validate model
if [[ ! " ${VALID_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "ERROR: Invalid model: $MODEL"
    echo "Valid options: ${VALID_MODELS[@]}"
    exit 1
fi

# Validate benchmarks
for B in "${BENCHMARKS[@]}"; do
    if [[ ! " ${VALID_BENCHMARKS[@]} " =~ " ${B} " ]]; then
        echo "ERROR: Invalid benchmark: $B"
        echo "Valid options: ${VALID_BENCHMARKS[@]} all"
        exit 1
    fi
done

# Expand & validate NFEs (only meaningful for flower_ablation and flower_cross_nfe)
if [[ "$MODEL" == "flower_ablation" ]]; then
    if [[ "$NFE_ARG" == "all" ]]; then
        NFES=("${ALL_NFES[@]}")
    else
        IFS=',' read -ra NFES <<<"$NFE_ARG"
    fi
    for N in "${NFES[@]}"; do
        if [[ ! " ${VALID_NFES[@]} " =~ " ${N} " ]]; then
            echo "ERROR: NFE must be 1, 2, or 3 (got $N)"
            exit 1
        fi
    done
elif [[ "$MODEL" == "flower_cross_nfe" ]]; then
    if [[ "$NFE_ARG" == "all" ]]; then
        NFES=("${ALL_NFES_CROSS[@]}")
    else
        IFS=',' read -ra NFES <<<"$NFE_ARG"
    fi
    for N in "${NFES[@]}"; do
        if [[ ! "$N" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: NFE must be a positive integer (got '$N')"
            exit 1
        fi
    done
else
    NFES=("")
fi

echo "Selected Benchmarks: ${BENCHMARKS[*]}"
echo "Selected Model:      $MODEL"
if [[ "$MODEL" == "flower_ablation" || "$MODEL" == "flower_cross_nfe" ]]; then
    echo "Selected NFE(s):     ${NFES[*]}"
fi
if [[ "$MODEL" == "flower_cross_nfe" ]]; then
    echo "Cross-NFE ckpt dir:  ${CROSS_NFE_CKPT_DIR:-<default: checkpoints/flower/\$BENCHMARK>}"
fi
echo "W&B logging:         $LOG_WANDB"
echo ""

# Run evaluation for one (benchmark, nfe) combination. Returns 0 on success, 1 on failure.
run_one_eval() {
    local BENCHMARK="$1"
    local NFE="$2"
    local NUM_SAMPLING_STEPS CHECKPOINT_DIR VARIANT_LABEL LOG_SUFFIX CKPT_FILE TRAIN_FOLDER LIBERO_DATASET_DIR

    # num_sampling_steps: matched to the NFE used at training for flower_ablation;
    # set to NFE for flower_cross_nfe; 4 for FLOWER baseline; 1 for single-step variants.
    if [[ "$MODEL" == "flower_ablation" || "$MODEL" == "flower_cross_nfe" ]]; then
        NUM_SAMPLING_STEPS="$NFE"
    elif [[ "$MODEL" == "flower" ]]; then
        NUM_SAMPLING_STEPS=4
    else
        NUM_SAMPLING_STEPS=1
    fi

    # Checkpoint directory: checkpoints/<model>/<benchmark>/ (or .../<benchmark>_nfe<N>/ for ablation).
    # flower_cross_nfe sweeps NFE on a single fixed ckpt (default: checkpoints/flower/<benchmark>/).
    if [[ "$MODEL" == "flower_ablation" ]]; then
        CHECKPOINT_DIR="${SCRIPT_DIR}/../checkpoints/flower_ablation/${BENCHMARK}_nfe${NFE}"
        VARIANT_LABEL="flower_nfe${NFE}"
        LOG_SUFFIX="flower_nfe${NFE}"
    elif [[ "$MODEL" == "flower_cross_nfe" ]]; then
        CHECKPOINT_DIR="${CROSS_NFE_CKPT_DIR:-${SCRIPT_DIR}/../checkpoints/flower/${BENCHMARK}}"
        VARIANT_LABEL="flower_cross_nfe${NFE}"
        LOG_SUFFIX="flower_cross_nfe${NFE}"
    else
        CHECKPOINT_DIR="${SCRIPT_DIR}/../checkpoints/${MODEL}/${BENCHMARK}"
        VARIANT_LABEL="$MODEL"
        LOG_SUFFIX="$MODEL"
    fi

    echo "------------------------------------------"
    if [[ "$MODEL" == "flower_ablation" || "$MODEL" == "flower_cross_nfe" ]]; then
        echo "Run: $BENCHMARK (NFE=$NFE)"
    else
        echo "Run: $BENCHMARK"
    fi
    echo "Checkpoint Directory: $CHECKPOINT_DIR"

    if [ ! -d "$CHECKPOINT_DIR" ] || [ ! "$(ls -A $CHECKPOINT_DIR)" ]; then
        echo "ERROR: Checkpoint not found at: $CHECKPOINT_DIR"
        echo ""
        echo "Expected structure:"
        echo "  $CHECKPOINT_DIR/"
        echo "    ├── .hydra/config.yaml"
        echo "    └── model.ckpt (or model.safetensors)"
        return 1
    fi

    # Auto-detect checkpoint file
    if [ -f "$CHECKPOINT_DIR/model.ckpt" ]; then
        CKPT_FILE="$CHECKPOINT_DIR/model.ckpt"
    elif [ -f "$CHECKPOINT_DIR/model.safetensors" ]; then
        CKPT_FILE="$CHECKPOINT_DIR/model.safetensors"
    else
        CKPT_FILE=$(find "$CHECKPOINT_DIR" -name "*.ckpt" -type f | head -1)
        if [ -z "$CKPT_FILE" ]; then
            echo "ERROR: No .ckpt or .safetensors file found in $CHECKPOINT_DIR"
            return 1
        fi
    fi

    # Auto-detect hydra config
    TRAIN_FOLDER="$CHECKPOINT_DIR/.hydra/config.yaml"
    if [ ! -f "$TRAIN_FOLDER" ]; then
        TRAIN_FOLDER=$(find "$CHECKPOINT_DIR" -path "*/.hydra/config.yaml" -type f | head -1)
        if [ -z "$TRAIN_FOLDER" ]; then
            echo "ERROR: .hydra/config.yaml not found in $CHECKPOINT_DIR"
            return 1
        fi
    fi

    LIBERO_DATASET_DIR="${SCRIPT_DIR}/../LIBERO/libero/datasets/${BENCHMARK}"
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

    python flower/evaluation/flower_eval_libero.py \
        train_folder="'$TRAIN_FOLDER'" \
        checkpoint="'$CKPT_FILE'" \
        benchmark_name="$BENCHMARK" \
        num_sampling_steps="$NUM_SAMPLING_STEPS" \
        eval_cfg_overwrite.model.num_sampling_steps="$NUM_SAMPLING_STEPS" \
        +eval_cfg_overwrite.model.load_pretrained=True \
        +variant_label="$VARIANT_LABEL" \
        log_dir="'${SCRIPT_DIR}/evaluation/${BENCHMARK}_${LOG_SUFFIX}_evaluation'" \
        wandb_entity=VLA-Thesis \
        device=0 \
        n_eval=20 \
        max_steps=520 \
        num_videos=5 \
        log_wandb="$LOG_WANDB"
}

# Sweep
SUCCEEDED_RUNS=()
FAILED_RUNS=()
for BENCHMARK in "${BENCHMARKS[@]}"; do
    if [[ "$MODEL" == "flower_ablation" || "$MODEL" == "flower_cross_nfe" ]]; then
        for NFE in "${NFES[@]}"; do
            LABEL="${BENCHMARK}/nfe${NFE}"
            if run_one_eval "$BENCHMARK" "$NFE"; then
                SUCCEEDED_RUNS+=("$LABEL")
                echo ""
                echo "OK  $LABEL"
            else
                FAILED_RUNS+=("$LABEL")
                echo ""
                echo "FAIL $LABEL"
            fi
        done
    else
        if run_one_eval "$BENCHMARK" ""; then
            SUCCEEDED_RUNS+=("$BENCHMARK")
            echo ""
            echo "OK  $BENCHMARK"
        else
            FAILED_RUNS+=("$BENCHMARK")
            echo ""
            echo "FAIL $BENCHMARK"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "Succeeded (${#SUCCEEDED_RUNS[@]}): ${SUCCEEDED_RUNS[*]:-<none>}"
if (( ${#FAILED_RUNS[@]} > 0 )); then
    echo "Failed    (${#FAILED_RUNS[@]}): ${FAILED_RUNS[*]}"
    echo ""
    echo "Results for successful runs saved under evaluation/<benchmark>_<variant>_evaluation/"
    exit 1
fi
echo ""
echo "All runs complete. Results saved under evaluation/<benchmark>_<variant>_evaluation/"
