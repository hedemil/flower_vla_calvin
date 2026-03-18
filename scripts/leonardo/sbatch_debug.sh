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
shift || true  # remaining args are hydra overrides

if [[ "$TASK" != "libero" && "$TASK" != "calvin" ]]; then
    echo "ERROR: First argument must be 'libero' or 'calvin', got: $TASK"
    exit 1
fi

# ---------------------
# Paths
# ---------------------
FAST="${LEONARDO_FAST:-${FAST:?Set LEONARDO_FAST or FAST}}"
WORK="${LEONARDO_WORK:-${WORK:?Set LEONARDO_WORK or WORK}}"

PROJECT_FAST="$FAST/flower_vla_calvin"
SIF="$WORK/containers/flower_vla_calvin.sif"

# ---------------------
# Environment variables
# ---------------------
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export WANDB_MODE=offline
export WANDB_DIR="$PROJECT_FAST/wandb_runs"

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
echo "Container:     $SIF"
echo "MASTER_ADDR:   $MASTER_ADDR"
echo "MASTER_PORT:   $MASTER_PORT"
echo "Hydra args:    $*"
echo "============================================"
echo ""

echo "--- GPU Info ---"
nvidia-smi
echo ""

echo "--- Container check ---"
if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Container not found: $SIF"
    exit 1
fi
echo "Container: $(ls -lh "$SIF")"
echo ""

# Run diagnostics inside container
singularity exec --nv \
    --no-home \
    --env HOME=/appuser \
    --bind "$WORK/hf_cache:/appuser/.cache/huggingface" \
    "$SIF" \
    bash -c '
echo "--- Python/PyTorch/CUDA ---"
python -c "
import sys; print(f\"Python: {sys.version}\")
import torch; print(f\"PyTorch: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
print(f\"CUDA version: {torch.version.cuda}\")
print(f\"GPU count: {torch.cuda.device_count()}\")
for i in range(torch.cuda.device_count()):
    print(f\"  GPU {i}: {torch.cuda.get_device_name(i)}\")
"
echo ""

echo "--- HuggingFace cache check ---"
python -c "
import os
cache = os.path.expanduser(\"~/.cache/huggingface\")
if os.path.exists(cache):
    total = sum(os.path.getsize(os.path.join(d,f)) for d,_,files in os.walk(cache) for f in files)
    print(f\"HF cache: {total / 1e9:.1f} GB\")
else:
    print(\"WARNING: HF cache not found!\")
"
echo ""

echo "--- Florence-2 load test ---"
python -c "
import os
os.environ[\"TRANSFORMERS_OFFLINE\"] = \"1\"
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained(\"microsoft/Florence-2-large\", trust_remote_code=True)
print(f\"Florence-2 processor loaded: {type(p).__name__}\")
" 2>&1 || echo "WARNING: Florence-2 load failed (run hf_cache setup first)"
'

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
    if [[ -d "$DATADIR/task_D_D/training/extracted" ]]; then
        echo "Extracted rel_actions: found"
    else
        echo "WARNING: Extracted rel_actions not found. Run preprocessing first."
    fi
fi
echo ""

# ---------------------
# Ensure wandb dir exists
# ---------------------
mkdir -p "$WANDB_DIR"

# ---------------------
# Build bind mounts and training command
# ---------------------
COMMON_BINDS=(
    --bind "$PROJECT_FAST/checkpoints:/workspace/flower_vla_calvin/checkpoints"
    --bind "$PROJECT_FAST/logs:/workspace/flower_vla_calvin/logs"
    --bind "$WORK/hf_cache:/appuser/.cache/huggingface"
    --bind "$PROJECT_FAST/conf:/workspace/flower_vla_calvin/conf"
    --bind "$PROJECT_FAST/flower:/workspace/flower_vla_calvin/flower"
    --bind "$PROJECT_FAST/wandb_runs:/workspace/flower_vla_calvin/wandb_runs"
)

COMMON_ENVS=(
    --env TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE
    --env HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE
    --env HF_HUB_OFFLINE=$HF_HUB_OFFLINE
    --env WANDB_MODE=$WANDB_MODE
    --env WANDB_DIR=/workspace/flower_vla_calvin/wandb_runs
    --env MUJOCO_GL=$MUJOCO_GL
    --env PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM
    --env NCCL_NET=$NCCL_NET
    --env NCCL_IB_HCA=$NCCL_IB_HCA
    --env NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL
    --env NCCL_DEBUG=$NCCL_DEBUG
    --env MASTER_ADDR=$MASTER_ADDR
    --env MASTER_PORT=$MASTER_PORT
    --env PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF
    --env CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER
    --env TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM
)

echo "--- Starting debug training ($TASK, 10 batches, 1 epoch) ---"

if [[ "$TASK" == "libero" ]]; then
    singularity exec --nv \
        --no-home \
        --env HOME=/appuser \
        "${COMMON_ENVS[@]}" \
        "${COMMON_BINDS[@]}" \
        --bind "$WORK/data/libero:/workspace/flower_vla_calvin/LIBERO/libero/datasets" \
        "$SIF" \
        python /workspace/flower_vla_calvin/flower/training_libero.py \
            devices=4 \
            log_dir=/workspace/flower_vla_calvin/logs \
            model=meanflower \
            trainer.limit_train_batches=10 \
            max_epochs=1 \
            rollout_lh_skip_epochs=9999 \
            "$@"
else
    singularity exec --nv \
        --no-home \
        --env HOME=/appuser \
        "${COMMON_ENVS[@]}" \
        "${COMMON_BINDS[@]}" \
        --bind "$WORK/data/calvin:/workspace/flower_vla_calvin/dataset" \
        "$SIF" \
        python /workspace/flower_vla_calvin/flower/training_calvin.py \
            devices=4 \
            log_dir=/workspace/flower_vla_calvin/logs \
            root_data_dir=/workspace/flower_vla_calvin/dataset/task_D_D \
            use_extracted_rel_actions=true \
            benchmark_name=calvin_d \
            model=meanflower \
            trainer.limit_train_batches=10 \
            max_epochs=1 \
            rollout_lh_skip_epochs=9999 \
            "$@"
fi

echo ""
echo "Debug job $SLURM_JOB_ID finished at $(date)"
