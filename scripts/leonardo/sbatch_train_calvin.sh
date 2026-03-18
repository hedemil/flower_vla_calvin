#!/bin/bash
# ==============================================================================
# CALVIN training on Leonardo HPC (CINECA)
#
# Usage:
#   sbatch scripts/leonardo/sbatch_train_calvin.sh [hydra overrides...]
#
# Examples:
#   sbatch scripts/leonardo/sbatch_train_calvin.sh
#   sbatch scripts/leonardo/sbatch_train_calvin.sh batch_size=8
#   sbatch scripts/leonardo/sbatch_train_calvin.sh benchmark_name=calvin_abcd
# ==============================================================================

#SBATCH --job-name=flower-calvin
#SBATCH --partition=boost_usr_prod
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ---------------------
# Paths
# ---------------------
FAST="${LEONARDO_FAST:-${FAST:?Set LEONARDO_FAST or FAST}}"
WORK="${LEONARDO_WORK:-${WORK:?Set LEONARDO_WORK or WORK}}"

PROJECT_FAST="$FAST/flower_vla_calvin"
SIF="$WORK/containers/flower_vla_calvin.sif"

# ---------------------
# Environment variables for compute node (no internet)
# ---------------------
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export WANDB_MODE=offline
export WANDB_DIR="$PROJECT_FAST/wandb_runs"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# NCCL for InfiniBand
export NCCL_NET=IB
export NCCL_IB_HCA=mlx5
export NCCL_NET_GDR_LEVEL=5

# DDP coordination
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=true

# ---------------------
# Logging
# ---------------------
echo "============================================"
echo "CALVIN Training - Leonardo HPC"
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPUs:          $SLURM_GPUS_ON_NODE"
echo "Container:     $SIF"
echo "MASTER_ADDR:   $MASTER_ADDR"
echo "MASTER_PORT:   $MASTER_PORT"
echo "Hydra args:    $*"
echo "============================================"

# ---------------------
# Ensure wandb dir exists
# ---------------------
mkdir -p "$WANDB_DIR"

# ---------------------
# Run training inside Singularity
# ---------------------
singularity exec --nv \
    --no-home \
    --env HOME=/appuser \
    --env TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE \
    --env HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE \
    --env HF_HUB_OFFLINE=$HF_HUB_OFFLINE \
    --env WANDB_MODE=$WANDB_MODE \
    --env WANDB_DIR=/workspace/flower_vla_calvin/wandb_runs \
    --env MUJOCO_GL=$MUJOCO_GL \
    --env PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM \
    --env NCCL_NET=$NCCL_NET \
    --env NCCL_IB_HCA=$NCCL_IB_HCA \
    --env NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    --env PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
    --env CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER \
    --env TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM \
    --bind "$WORK/data/calvin:/workspace/flower_vla_calvin/dataset" \
    --bind "$PROJECT_FAST/checkpoints:/workspace/flower_vla_calvin/checkpoints" \
    --bind "$PROJECT_FAST/logs:/workspace/flower_vla_calvin/logs" \
    --bind "$WORK/hf_cache:/appuser/.cache/huggingface" \
    --bind "$PROJECT_FAST/conf:/workspace/flower_vla_calvin/conf" \
    --bind "$PROJECT_FAST/flower:/workspace/flower_vla_calvin/flower" \
    --bind "$PROJECT_FAST/wandb_runs:/workspace/flower_vla_calvin/wandb_runs" \
    "$SIF" \
    python /workspace/flower_vla_calvin/flower/training_calvin.py \
        devices=4 \
        log_dir=/workspace/flower_vla_calvin/logs \
        root_data_dir=/workspace/flower_vla_calvin/dataset/task_D_D \
        use_extracted_rel_actions=true \
        benchmark_name=calvin_d \
        model=meanflower \
        "$@"

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
