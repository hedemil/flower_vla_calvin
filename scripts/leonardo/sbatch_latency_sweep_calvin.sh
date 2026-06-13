#!/bin/bash
# ==============================================================================
# Decoupled-clock latency sweep on a fine-tuned CALVIN checkpoint (Experiment F).
#
# CALVIN sim pauses the world while the policy runs, so inference latency is
# invisible to the success rate. This sweep re-couples the clocks: with
# inference_delay_steps=d the action applied at control step k is the one the
# policy computed d steps earlier, emulating a real robot acting on a stale
# observation while the network thinks. Sweeping d turns the (otherwise hidden)
# latency advantage of the 1-step iMF head into a measurable success-rate gap.
#
# Map a measured latency L (ms) to d = round(L / 33.3) (CALVIN is 30 Hz).
# RTX 4090 operating points: iMF ~32.9 ms -> d=1, RF ~67.0 ms -> d=2.
#
# Scope (per the approved plan): one checkpoint (tuned iMF), D->D, 100 sequences.
# Mirrors scripts/leonardo/sbatch_nfe_sweep_calvin.sh.
#
# Usage:
#   sbatch scripts/leonardo/sbatch_latency_sweep_calvin.sh <run_dir> [hydra overrides...]
#
# Env vars:
#   DELAYS="0 1 2 3 4 5 6 8"   # space-separated control-step delays to sweep
#   NUM_SEQUENCES=100          # 100 for the reduced sweep, 1000 for full power
#   NFE=1                      # sampling steps (1 for iMF; set 4 for an RF ckpt)
#
# After the run, aggregate + plot with:
#   python -m tools.results.calvin_latency_sweep \
#       --sweep-root <run_dir>/latency_sweep \
#       --out-dir docs/figures/results/realtime
# ==============================================================================

#SBATCH --job-name=latency-sweep-calvin
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

DELAYS="${DELAYS:-0 1 2 3 4 5 6 8}"
NUM_SEQUENCES="${NUM_SEQUENCES:-100}"
NFE="${NFE:-1}"

FAST="${LEONARDO_FAST:-${FAST:?Set LEONARDO_FAST or FAST}}"
WORK="${LEONARDO_WORK:-${WORK:?Set LEONARDO_WORK or WORK}}"

CODE_DIR="$FAST/project/flower_vla_calvin"
VENV_DIR="$WORK/venvs/flower_vla_calvin"
HF_CACHE="$WORK/ehed0000/hf_cache"

# Locate train_folder and best ckpt (EMA safetensors preferred, as in nfe sweep).
TRAIN_FOLDER="$RUN_DIR/.hydra/config.yaml"
[[ -f "$TRAIN_FOLDER" ]] || { echo "ERROR: $TRAIN_FOLDER not found"; exit 1; }

CKPT="$(find "$RUN_DIR" -path "*/saved_models/*/model.safetensors" | head -n 1)"
if [[ -z "$CKPT" ]]; then
    echo "INFO: no EMA model.safetensors found, falling back to raw-weight .ckpt"
    CKPT="$(find "$RUN_DIR" -path "*/saved_models/*.ckpt" | head -n 1)"
fi
[[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint found under $RUN_DIR"; exit 1; }
echo "Selected checkpoint: $CKPT"

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
echo "CALVIN latency (decoupled-clock) sweep on $(basename "$RUN_DIR")"
echo "============================================"
echo "Checkpoint:     $CKPT"
echo "Data:           $DATA_DIR"
echo "Delays:         $DELAYS"
echo "num_sequences:  $NUM_SEQUENCES"
echo "NFE:            $NFE"
echo "Start:          $(date)"
echo "============================================"

cd "$CODE_DIR"

for D in $DELAYS; do
    EVAL_OUT="$RUN_DIR/latency_sweep/d${D}"
    mkdir -p "$EVAL_OUT"
    echo ""
    echo "=== inference_delay_steps=$D  ->  $EVAL_OUT  ($(date)) ==="
    # log_actions=true so smoothness-under-delay can also be computed.
    # See sbatch_nfe_sweep_calvin.sh for why checkpoint/train_folder are quoted
    # and num_videos=0.
    python flower/evaluation/flower_evaluate.py \
        "train_folder='$TRAIN_FOLDER'" \
        "checkpoint='$CKPT'" \
        log_dir="$EVAL_OUT" \
        "dataset_path='$DATA_DIR'" \
        num_sequences="$NUM_SEQUENCES" \
        num_sampling_steps="$NFE" \
        inference_delay_steps="$D" \
        log_actions=true \
        num_videos=0 \
        log_wandb=False \
        "$@"
    echo "d=$D done at $(date)"
done

echo ""
echo "=== latency sweep complete at $(date) ==="
echo "Per-delay JSONLs under:  $RUN_DIR/latency_sweep/d*/logs/*/rollout_episodes.jsonl"
echo "Aggregate + plot:        python -m tools.results.calvin_latency_sweep \\"
echo "                             --sweep-root $RUN_DIR/latency_sweep \\"
echo "                             --out-dir docs/figures/results/realtime"
