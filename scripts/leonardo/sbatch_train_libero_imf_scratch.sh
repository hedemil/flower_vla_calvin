#!/bin/bash
# ==============================================================================
# LIBERO iMF from-scratch ablation (no pretrained checkpoint)
#
# Trains iMF on LIBERO-10 from random initialization under one of four
# parameter variants. Removes the pretrain-mismatch confound that affects
# the finetune-time ablations.
#
# Variants:
#   default - model.ratio=0.25, model.imf_head_depth=8 (config defaults)
#   ratio   - model.ratio=0.5
#   heads   - model.imf_head_depth=12
#   both    - model.ratio=0.5 + model.imf_head_depth=12
#
# Usage:
#   sbatch scripts/leonardo/sbatch_train_libero_imf_scratch.sh <variant> [hydra overrides...]
#
# Example:
#   sbatch scripts/leonardo/sbatch_train_libero_imf_scratch.sh ratio
# ==============================================================================

#SBATCH --job-name=imf-libero-scratch
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2-12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=AIFAC_F02_024

set -euo pipefail

# ---------------------
# Variant dispatch
# ---------------------
VARIANT="${1:?Usage: sbatch $0 <default|ratio|heads|both> [hydra overrides...]}"
shift

case "$VARIANT" in
    default) ABLATION_OVERRIDES=() ;;
    ratio)   ABLATION_OVERRIDES=(model.ratio=0.5) ;;
    heads)   ABLATION_OVERRIDES=(model.imf_head_depth=12) ;;
    both)    ABLATION_OVERRIDES=(model.ratio=0.5 model.imf_head_depth=12) ;;
    *)
        echo "ERROR: Unknown variant '$VARIANT'. Expected: default | ratio | heads | both"
        exit 1
        ;;
esac

# ---------------------
# Paths
# ---------------------
FAST="${LEONARDO_FAST:-${FAST:?Set LEONARDO_FAST or FAST}}"
WORK="${LEONARDO_WORK:-${WORK:?Set LEONARDO_WORK or WORK}}"

CODE_DIR="$FAST/project/flower_vla_calvin"
VENV_DIR="$WORK/venvs/flower_vla_calvin"
DATA_DIR="$WORK/data/libero"
HF_CACHE="$WORK/ehed0000/hf_cache"
WANDB_DIR="$CODE_DIR/wandb_runs"

# WandB run name
MODEL="imf"
DATASET="libero_10"
DATE=$(date +%Y%m%d)
WANDB_NAME="imf_${DATASET}_scratch_${VARIANT}_${DATE}"

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
echo "LIBERO iMF From-Scratch Ablation - Leonardo HPC"
echo "Variant: $VARIANT"
echo "Overrides: ${ABLATION_OVERRIDES[*]:-<none>}"
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPUs:          $SLURM_GPUS_ON_NODE"
echo "Model:         $MODEL (iMF, random init)"
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
echo "Hydra args:    ${ABLATION_OVERRIDES[*]:-<none>} $*"
echo "============================================"

mkdir -p "$WANDB_DIR"

# ---------------------
# Launch training (no +pretrain_chk: random init)
# ---------------------
cd "$CODE_DIR"

srun python flower/training_libero.py \
    devices=4 \
    log_dir="$CODE_DIR/logs" \
    root_data_dir="$DATA_DIR/$DATASET" \
    libero_benchmark="$DATASET" \
    model="$MODEL" \
    "${ABLATION_OVERRIDES[@]}" \
    logger.name="$WANDB_NAME" \
    hydra.run.dir="$CODE_DIR/logs/runs/\${now:%Y-%m-%d}/imf_scratch_${VARIANT}_${SLURM_JOB_ID}" \
    +callbacks.checkpoint.save_weights_only=True \
    "$@"

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
