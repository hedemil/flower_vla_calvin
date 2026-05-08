#!/bin/bash
# ==============================================================================
# LIBERO iMF ablation eval - Leonardo HPC sbatch wrapper.
#
# Single-GPU eval of one (variant, suite) combination. Submit one job per
# combination; 4 variants x 2 suites = 8 jobs.
#
# Usage:
#   sbatch scripts/leonardo/sbatch_libero_imf_ablation.sh <variant> <suite>
#
#   variant: default | ratio | heads | both
#   suite:   libero_10 | libero_spatial
#
# Example - submit all 8 jobs in parallel:
#   for v in default ratio heads both; do
#     for s in libero_10 libero_spatial; do
#       sbatch scripts/leonardo/sbatch_libero_imf_ablation.sh "$v" "$s"
#     done
#   done
# ==============================================================================

#SBATCH --job-name=imf-libero-eval
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=AIFAC_F02_024

set -euo pipefail

VARIANT="${1:?Usage: sbatch $0 <default|ratio|heads|both> <libero_10|libero_spatial>}"
SUITE="${2:?Usage: sbatch $0 <default|ratio|heads|both> <libero_10|libero_spatial>}"

case "$VARIANT" in
    default|ratio|heads|both) ;;
    *)
        echo "ERROR: Unknown variant '$VARIANT'. Expected: default | ratio | heads | both"
        exit 1
        ;;
esac

case "$SUITE" in
    libero_10|libero_spatial) ;;
    *)
        echo "ERROR: Unknown suite '$SUITE'. Expected: libero_10 | libero_spatial"
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
echo "LIBERO iMF Ablation Eval - Leonardo HPC"
echo "============================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "GPUs:        $SLURM_GPUS_ON_NODE"
echo "Variant:     imf_$VARIANT"
echo "Suite:       $SUITE"
echo "Code:        $CODE_DIR"
echo "Venv:        $VENV_DIR"
echo "Data:        $DATA_DIR"
echo "Python:      $(which python)"
echo "PyTorch:     $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:        $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"

mkdir -p "$WANDB_DIR"

cd "$CODE_DIR"

bash scripts/run_libero_imf_ablation.sh "$VARIANT" "$SUITE"

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
