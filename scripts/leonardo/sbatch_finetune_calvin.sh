#!/bin/bash
# ==============================================================================
# CALVIN paired fine-tune on Leonardo HPC.
#
# Defaults to CALVIN D (task_D_D) — single-environment, smaller download,
# largest FLOWER margin in Table 9 (+10pp over 2nd best, so unsaturated and
# the most likely place for an objective change to show up).
#
# To run on CALVIN ABC->D instead, set:
#   CALVIN_TASK=task_ABC_D BENCHMARK_NAME=calvin_abcd sbatch ... <model>
#
# Both runs MUST use the same seed and same pretrained checkpoint so that
# (initial_state, eval_sequence) pairs match for paired McNemar later.
#
# Usage:
#   sbatch scripts/leonardo/sbatch_finetune_calvin.sh <model> [hydra overrides...]
#
# Examples:
#   sbatch scripts/leonardo/sbatch_finetune_calvin.sh flower
#   sbatch scripts/leonardo/sbatch_finetune_calvin.sh imf
#   sbatch scripts/leonardo/sbatch_finetune_calvin.sh imf max_epochs=20
# ==============================================================================

#SBATCH --job-name=ft-calvin
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=AIFAC_F02_024

set -euo pipefail

# ---------------------
# Parse model argument
# ---------------------
MODEL="${1:?usage: $0 <model> [hydra overrides...]   (model is flower | imf)}"
shift 1

# ---------------------
# Paths
# ---------------------
FAST="${LEONARDO_FAST:-${FAST:?Set LEONARDO_FAST or FAST}}"
WORK="${LEONARDO_WORK:-${WORK:?Set LEONARDO_WORK or WORK}}"

CODE_DIR="$FAST/project/flower_vla_calvin"
VENV_DIR="$WORK/venvs/flower_vla_calvin"
CALVIN_TASK="${CALVIN_TASK:-task_D_D}"
BENCHMARK_NAME="${BENCHMARK_NAME:-calvin_d}"
DATA_DIR="$WORK/data/calvin/$CALVIN_TASK"
HF_CACHE="$WORK/ehed0000/hf_cache"
WANDB_DIR="$CODE_DIR/wandb_runs"

# Pretrained checkpoint per model (override with PRETRAIN_CHK env, or "" to disable).
# Mirrors the convention in sbatch_train_libero.sh / sbatch_train_libero_imf.sh:
# each model starts from its own matched pretrained backbone, so RF+RF-pretraining
# is compared end-to-end against iMF+iMF-pretraining.
if [[ -z "${PRETRAIN_CHK+x}" ]]; then
    case "$MODEL" in
        flower)
            PRETRAIN_CHK="$WORK/checkpoints/pretrained/flower_baseline_290000.safetensors"
            ;;
        imf)
            PRETRAIN_CHK="$WORK/checkpoints/pretrained/imf_checkpoint_290000.safetensors"
            ;;
        *)
            PRETRAIN_CHK=""
            ;;
    esac
fi

# Deterministic seed shared by RF and iMF for paired episode matching.
SEED="${CALVIN_FT_SEED:-242}"

DATE=$(date +%Y%m%d)
WANDB_NAME="${MODEL}_${BENCHMARK_NAME}_ft_seed${SEED}_${DATE}"

# ---------------------
# Modules and venv
# ---------------------
module purge
module load profile/deeplrn
module load cuda/12.1
source "$VENV_DIR/bin/activate"

# ---------------------
# Offline env (no internet on compute nodes)
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

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=true

# ---------------------
# Prerequisites
# ---------------------
[[ -d "$VENV_DIR" ]] || { echo "ERROR: Venv not found: $VENV_DIR"; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "ERROR: CALVIN data not found: $DATA_DIR (CALVIN_TASK=$CALVIN_TASK)"; exit 1; }
if [[ -n "$PRETRAIN_CHK" && ! -f "$PRETRAIN_CHK" ]]; then
    echo "ERROR: PRETRAIN_CHK not found: $PRETRAIN_CHK"; exit 1
fi

# ---------------------
# Logging
# ---------------------
echo "============================================"
echo "CALVIN $BENCHMARK_NAME fine-tune  ($MODEL)"
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPUs:          $SLURM_GPUS_ON_NODE"
echo "Model:         $MODEL"
echo "Seed:          $SEED"
echo "Benchmark:     $BENCHMARK_NAME ($CALVIN_TASK)"
echo "Pretrain ckpt: ${PRETRAIN_CHK:-<none>}"
echo "Data:          $DATA_DIR"
echo "WandB name:    $WANDB_NAME"
echo "MASTER_ADDR:   $MASTER_ADDR"
echo "MASTER_PORT:   $MASTER_PORT"
echo "Hydra args:    $*"
echo "============================================"

mkdir -p "$WANDB_DIR"

cd "$CODE_DIR"

# ---------------------
# Launch
# ---------------------
# Per-model strict_load convention (mirrors sbatch_train_libero*.sh):
#   - RF: strict_load=true (no key remap; loud failure if mismatch)
#   - iMF: strict_load left at default false (allows dit.N -> shared+u/v head remap)
EXTRA_ARGS=()
if [[ -n "$PRETRAIN_CHK" ]]; then
    EXTRA_ARGS+=("+pretrain_chk=$PRETRAIN_CHK")
    if [[ "$MODEL" == "flower" ]]; then
        EXTRA_ARGS+=("+strict_load=true")
    fi
fi

# Fine-tuning regime overrides (max_epochs/skip/freq) are set here so RF and
# iMF runs use identical schedules out of the box. Override on the CLI to
# experiment. Eval starts at epoch 1 (skip_epochs=1) and runs every 2 epochs
# so we get ~10+ eval points across a 25-epoch fine-tune.
srun python flower/training_calvin.py \
    devices=4 \
    seed="$SEED" \
    log_dir="$CODE_DIR/logs" \
    root_data_dir="$DATA_DIR" \
    use_extracted_rel_actions=true \
    benchmark_name="$BENCHMARK_NAME" \
    model="$MODEL" \
    max_epochs=25 \
    rollout_lh_skip_epochs=1 \
    callbacks.rollout_lh.rollout_freq=2 \
    logger.name="$WANDB_NAME" \
    hydra.run.dir="$CODE_DIR/logs/runs/\${now:%Y-%m-%d}/${MODEL}_${BENCHMARK_NAME}_${SLURM_JOB_ID}" \
    +callbacks.checkpoint.save_weights_only=True \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
