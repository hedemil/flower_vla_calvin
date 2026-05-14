#!/bin/bash
# ==============================================================================
# NFE sweep on a fine-tuned RF (FLOWER) CALVIN checkpoint.
#
# Loads the best checkpoint from a CALVIN fine-tune run and re-evaluates at
# NFE = 1, 2, 3, 4 sequentially. Per-sequence JSONL is written for each NFE.
# Mirrors the libero_10_flower_cross_nfeN_evaluation protocol from your LIBERO
# analysis — directly answers "does RF saturate at NFE=1 on CALVIN like it did
# on LIBERO?".
#
# Only applies to RF (flower). iMF's sample_actions is hardcoded to NFE=1.
#
# Usage:
#   sbatch scripts/leonardo/sbatch_nfe_sweep_calvin.sh <run_dir> [hydra overrides...]
#
# Env vars:
#   NFES="1 2 3 4"            # space-separated list of NFE values to sweep
#   NUM_SEQUENCES=300         # 300 for fast, 1000 for paper-comparable
#
# Example:
#   sbatch scripts/leonardo/sbatch_nfe_sweep_calvin.sh \
#     $LEONARDO_FAST/project/flower_vla_calvin/logs/runs/2026-05-13/flower_calvin_d_41440220
# ==============================================================================

#SBATCH --job-name=nfe-sweep-calvin
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=AIFAC_F02_024

set -euo pipefail

RUN_DIR="${1:?usage: $0 <run_dir> [hydra overrides...]}"
shift 1
RUN_DIR="$(readlink -f "$RUN_DIR")"

NFES="${NFES:-1 2 3 4}"
NUM_SEQUENCES="${NUM_SEQUENCES:-300}"

FAST="${LEONARDO_FAST:-${FAST:?Set LEONARDO_FAST or FAST}}"
WORK="${LEONARDO_WORK:-${WORK:?Set LEONARDO_WORK or WORK}}"

CODE_DIR="$FAST/project/flower_vla_calvin"
VENV_DIR="$WORK/venvs/flower_vla_calvin"
HF_CACHE="$WORK/ehed0000/hf_cache"

# Locate train_folder and best ckpt
TRAIN_FOLDER="$RUN_DIR/.hydra/config.yaml"
[[ -f "$TRAIN_FOLDER" ]] || { echo "ERROR: $TRAIN_FOLDER not found"; exit 1; }

# Prefer EMA safetensors (HF format dir with model.safetensors + config.yaml)
# over the raw-weight .ckpt — gives EMA-eval numbers comparable to wandb.
CKPT="$(find "$RUN_DIR" -path "*/saved_models/*/model.safetensors" | head -n 1)"
if [[ -z "$CKPT" ]]; then
    echo "INFO: no EMA model.safetensors found, falling back to raw-weight .ckpt"
    CKPT="$(find "$RUN_DIR" -path "*/saved_models/*.ckpt" | head -n 1)"
fi
[[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint found under $RUN_DIR"; exit 1; }
echo "Selected checkpoint: $CKPT"

# Pull benchmark + data path from the training config
BENCHMARK_NAME="$(grep '^benchmark_name:' "$TRAIN_FOLDER" | awk '{print $2}')"
DATA_DIR="$(grep '^root_data_dir:' "$TRAIN_FOLDER" | awk '{print $2}')"
[[ -d "$DATA_DIR" ]] || { echo "ERROR: data dir $DATA_DIR (from $TRAIN_FOLDER) not found"; exit 1; }

# Env
module purge
module load profile/deeplrn
module load cuda/12.1
source "$VENV_DIR/bin/activate"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="$HF_CACHE"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=true

echo "============================================"
echo "CALVIN NFE sweep on $(basename "$RUN_DIR")"
echo "============================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Run dir:        $RUN_DIR"
echo "Benchmark:      $BENCHMARK_NAME"
echo "Checkpoint:     $CKPT"
echo "Data:           $DATA_DIR"
echo "NFEs:           $NFES"
echo "num_sequences:  $NUM_SEQUENCES"
echo "Hydra args:     $*"
echo "Start:          $(date)"
echo "============================================"

cd "$CODE_DIR"

for NFE in $NFES; do
    EVAL_OUT="$RUN_DIR/nfe_sweep/nfe_${NFE}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$EVAL_OUT"
    echo ""
    echo "============================================"
    echo "NFE=$NFE  →  $EVAL_OUT"
    echo "Start: $(date)"
    echo "============================================"
    # NOTE: checkpoint path contains '=' signs from PL's filename template
    # (e.g. .../epoch=69_eval_lh/avg_seq_len=4.23.ckpt). Hydra's override grammar
    # would parse the embedded '=' as additional key=value separators. Quoting
    # the value with single quotes tells Hydra to treat the whole thing as a
    # string literal. train_folder also gets the same treatment defensively.
    # num_videos=0: eval_calvin.yaml defaults to 30, which triggers wandb video
    # logging that requires moviepy+imageio (not installed) and crashes the run
    # AFTER all rollouts complete, losing the JSONL. We don't need videos.
    python flower/evaluation/flower_evaluate.py \
        "train_folder='$TRAIN_FOLDER'" \
        "checkpoint='$CKPT'" \
        log_dir="$EVAL_OUT" \
        "dataset_path='$DATA_DIR'" \
        num_sequences="$NUM_SEQUENCES" \
        num_sampling_steps="$NFE" \
        num_videos=0 \
        log_wandb=False \
        "$@"
    echo "NFE=$NFE done at $(date)"
done

echo ""
echo "=== NFE sweep complete at $(date) ==="
echo "Per-NFE JSONLs under:  $RUN_DIR/nfe_sweep/nfe_*/logs/*/rollout_episodes.jsonl"
