#!/bin/bash
# ==============================================================================
# LIBERO training on Leonardo HPC (venv-based, no container)
#
# Usage:
#   sbatch scripts/leonardo/sbatch_train_libero.sh <model> [hydra overrides...]
#
# Examples:
#   sbatch scripts/leonardo/sbatch_train_libero.sh meanflower
#   sbatch scripts/leonardo/sbatch_train_libero.sh decoupled_meanflower batch_size=8
#   sbatch scripts/leonardo/sbatch_train_libero.sh meanflower libero_benchmark=libero_goal
# ==============================================================================

#SBATCH --job-name=flower-libero
#SBATCH --partition=boost_usr_prod
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
# Parse model argument
# ---------------------
MODEL="${1:-meanflower}"
shift 1 || true

# ---------------------
# Paths
# ---------------------
FAST="${LEONARDO_FAST:-${FAST:?Set LEONARDO_FAST or FAST}}"
WORK="${LEONARDO_WORK:-${WORK:?Set LEONARDO_WORK or WORK}}"

CODE_DIR="$FAST/flower_vla_calvin"
VENV_DIR="$WORK/venvs/flower_vla_calvin"
DATA_DIR="$WORK/data/libero"
HF_CACHE="$WORK/ehed0000/hf_cache"
WANDB_DIR="$CODE_DIR/wandb_runs"

# WandB run name: model_dataset_date
DATASET="libero_spatial"
DATE=$(date +%Y%m%d)
WANDB_NAME="${MODEL}_${DATASET}_${DATE}"

# ---------------------
# Load modules and activate venv
# ---------------------
module purge
module load profile/deeplrn
module load cuda/12.1
source "$VENV_DIR/bin/activate"

# ---------------------
# Environment variables (no internet on compute nodes)
# ---------------------
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="$HF_CACHE"

export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"

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
# Verify prerequisites
# ---------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Venv not found: $VENV_DIR"
    echo "  Run setup_leonardo.sh first."
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: LIBERO data not found: $DATA_DIR"
    exit 1
fi

# ---------------------
# Logging
# ---------------------
echo "============================================"
echo "LIBERO Training - Leonardo HPC"
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPUs:          $SLURM_GPUS_ON_NODE"
echo "Model:         $MODEL"
echo "Dataset:       $DATASET"
echo "WandB name:    $WANDB_NAME"
echo "Code:          $CODE_DIR"
echo "Venv:          $VENV_DIR"
echo "Data:          $DATA_DIR"
echo "MASTER_ADDR:   $MASTER_ADDR"
echo "MASTER_PORT:   $MASTER_PORT"
echo "Python:        $(which python)"
echo "PyTorch:       $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:          $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Hydra args:    $*"
echo "============================================"

mkdir -p "$WANDB_DIR"

# ---------------------
# Launch training
# ---------------------
cd "$CODE_DIR"

cd "$CODE_DIR"

python flower/training_libero.py \
    devices=4 \
    log_dir="$CODE_DIR/logs" \
    root_data_dir="$DATA_DIR/$DATASET" \
    model="$MODEL" \
    logger.name="$WANDB_NAME" \
    hydra.run.dir="$CODE_DIR/logs/runs/\${now:%Y-%m-%d}/${MODEL}_${SLURM_JOB_ID}" \
    "$@"

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
