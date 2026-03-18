#!/bin/bash
# ==============================================================================
# Debug job for Leonardo HPC - quick validation (30 min)
#
# Usage:
#   sbatch scripts/leonardo/sbatch_debug.sh libero [hydra overrides...]
#   sbatch scripts/leonardo/sbatch_debug.sh calvin [hydra overrides...]
# ==============================================================================

#SBATCH --job-name=flower-debug
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ---------------------
# Parse task argument
# ---------------------
TASK="${1:-libero}"
shift || true

if [[ "$TASK" != "libero" && "$TASK" != "calvin" ]]; then
    echo "ERROR: First argument must be 'libero' or 'calvin', got: $TASK"
    exit 1
fi

# ---------------------
# Paths
# ---------------------
FAST="${LEONARDO_FAST:-${FAST:?Set LEONARDO_FAST or FAST}}"
WORK="${LEONARDO_WORK:-${WORK:?Set LEONARDO_WORK or WORK}}"

CODE_DIR="$FAST/flower_vla_calvin"
VENV_DIR="$WORK/venvs/flower_vla_calvin"
HF_CACHE="$WORK/hf_cache"
WANDB_DIR="$CODE_DIR/wandb_runs"

# ---------------------
# Load modules and activate venv
# ---------------------
module purge
module load profile/deeplrn
module load cuda/12.1
source "$VENV_DIR/bin/activate"

# ---------------------
# Environment variables
# ---------------------
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="$HF_CACHE"

export WANDB_MODE=offline
export WANDB_DIR="$WANDB_DIR"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export NCCL_NET=IB
export NCCL_IB_HCA=mlx5
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=INFO

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=true

# ---------------------
# Diagnostics
# ---------------------
echo "============================================"
echo "DEBUG Job - Leonardo HPC"
echo "============================================"
echo "Task:          $TASK"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPUs:          $SLURM_GPUS_ON_NODE"
echo "Code:          $CODE_DIR"
echo "Venv:          $VENV_DIR"
echo "MASTER_ADDR:   $MASTER_ADDR"
echo "MASTER_PORT:   $MASTER_PORT"
echo "Hydra args:    $*"
echo "============================================"
echo ""

echo "--- GPU Info ---"
nvidia-smi
echo ""

echo "--- Python/PyTorch/CUDA ---"
python -c "
import sys; print(f'Python: {sys.version}')
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

echo "--- HuggingFace cache check ---"
python -c "
import os
cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
if os.path.exists(cache):
    total = sum(os.path.getsize(os.path.join(d,f)) for d,_,files in os.walk(cache) for f in files)
    print(f'HF cache: {total / 1e9:.1f} GB at {cache}')
else:
    print(f'WARNING: HF cache not found at {cache}')
"
echo ""

echo "--- Florence-2 load test ---"
python -c "
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)
print(f'Florence-2 processor loaded: {type(p).__name__}')
" 2>&1 || echo "WARNING: Florence-2 load failed (run setup_leonardo.sh hf_cache first)"
echo ""

# ---------------------
# Data directory checks
# ---------------------
echo "--- Data directory checks ---"
if [[ "$TASK" == "libero" ]]; then
    DATADIR="$WORK/data/libero"
    echo "LIBERO data: $DATADIR"
    ls -la "$DATADIR/" 2>/dev/null || echo "WARNING: LIBERO data dir empty"
else
    DATADIR="$WORK/data/calvin"
    echo "CALVIN data: $DATADIR"
    ls -la "$DATADIR/" 2>/dev/null || echo "WARNING: CALVIN data dir empty"
    if [[ -d "$DATADIR/task_ABC_D/training/extracted" ]]; then
        echo "Extracted rel_actions: found"
    else
        echo "WARNING: Extracted rel_actions not found. Run preprocessing first."
    fi
fi
echo ""

mkdir -p "$WANDB_DIR"

# ---------------------
# Debug training (10 batches, 1 epoch)
# ---------------------
echo "--- Starting debug training ($TASK, 10 batches, 1 epoch) ---"
cd "$CODE_DIR"

if [[ "$TASK" == "libero" ]]; then
    python flower/training_libero.py \
        devices=4 \
        log_dir="$CODE_DIR/logs" \
        root_data_dir="$WORK/data/libero/libero_spatial" \
        model=meanflower \
        trainer.limit_train_batches=10 \
        max_epochs=1 \
        rollout_lh_skip_epochs=9999 \
        "$@"
else
    python flower/training_calvin.py \
        devices=4 \
        log_dir="$CODE_DIR/logs" \
        root_data_dir="$WORK/data/calvin/task_ABC_D" \
        use_extracted_rel_actions=true \
        benchmark_name=calvin_abcd \
        model=meanflower \
        trainer.limit_train_batches=10 \
        max_epochs=1 \
        rollout_lh_skip_epochs=9999 \
        "$@"
fi

echo ""
echo "Debug job $SLURM_JOB_ID finished at $(date)"
