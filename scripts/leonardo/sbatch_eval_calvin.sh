#!/bin/bash
# ==============================================================================
# Post-training CALVIN evaluation at num_sequences=1000 (Table 10-comparable).
#
# Run AFTER sbatch_finetune_calvin.sh completes. Loads the best checkpoint
# from a fine-tune run, replays 1000 chains, and writes per-sequence JSONL.
#
# Usage:
#   sbatch scripts/leonardo/sbatch_eval_calvin.sh <run_dir>  [hydra overrides...]
#
# <run_dir> is the Hydra run dir of the fine-tune job, e.g.:
#   $LEONARDO_FAST/project/flower_vla_calvin/logs/runs/2026-05-13/imf_calvin_d_12345
#
# The script auto-detects train_folder (run_dir/.hydra/config.yaml) and the
# best checkpoint (highest eval_lh/avg_seq_len) under run_dir/seed_*/saved_models/.
#
# Output:
#   $log_dir/logs/<timestamp>/rollout_episodes.jsonl    (per-sequence records)
#   $log_dir/logs/<timestamp>/results.json              (aggregate chain SR)
# ==============================================================================

#SBATCH --job-name=eval-calvin
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

FAST="${LEONARDO_FAST:-${FAST:?Set LEONARDO_FAST or FAST}}"
WORK="${LEONARDO_WORK:-${WORK:?Set LEONARDO_WORK or WORK}}"

CODE_DIR="$FAST/project/flower_vla_calvin"
VENV_DIR="$WORK/venvs/flower_vla_calvin"
HF_CACHE="$WORK/ehed0000/hf_cache"

# Locate train_folder (Hydra config snapshot)
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

# Pick CALVIN benchmark and dataset path from the fine-tune config
BENCHMARK_NAME="$(grep '^benchmark_name:' "$TRAIN_FOLDER" | awk '{print $2}')"
DATA_DIR="$(grep '^root_data_dir:' "$TRAIN_FOLDER" | awk '{print $2}')"
[[ -d "$DATA_DIR" ]] || { echo "ERROR: data dir $DATA_DIR (from $TRAIN_FOLDER) not found"; exit 1; }

# Timestamped output dir under the run dir
EVAL_OUT="$RUN_DIR/final_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_OUT"

# ---------------------
# Env
# ---------------------
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
echo "CALVIN final eval (num_sequences=1000)"
echo "============================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Run dir:        $RUN_DIR"
echo "Benchmark:      $BENCHMARK_NAME"
echo "Train folder:   $TRAIN_FOLDER"
echo "Checkpoint:     $CKPT"
echo "Data:           $DATA_DIR"
echo "Eval out:       $EVAL_OUT"
echo "Hydra args:     $*"
echo "============================================"

cd "$CODE_DIR"

# Note: flower_evaluate.py uses Hydra config eval_calvin.yaml by default.
# We override train_folder/checkpoint/log_dir/num_sequences/dataset_path.
# Single-quoted values shield embedded '=' signs (PL filename pattern
# epoch=NN_metric=X.XX.ckpt) from Hydra's override grammar parser.
# num_videos=0: avoid wandb video-logging crash that needs moviepy+imageio.
python flower/evaluation/flower_evaluate.py \
    "train_folder='$TRAIN_FOLDER'" \
    "checkpoint='$CKPT'" \
    log_dir="$EVAL_OUT" \
    "dataset_path='$DATA_DIR'" \
    num_sequences=1000 \
    num_videos=0 \
    log_wandb=False \
    "$@"

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Per-sequence JSONL: $EVAL_OUT/logs/<timestamp>/rollout_episodes.jsonl"
