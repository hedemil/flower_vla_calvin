#!/bin/bash
# ==============================================================================
# Download a CALVIN EMA checkpoint from the HF Hub and run the headline
# num_sequences=1000 evaluation on an RTX 4090 box (run via ssh, no slurm).
#
# This is the 4090 half of the Leonardo -> HF -> 4090 workflow. Leonardo
# uploads with scripts/leonardo/upload_checkpoint_hf.py; this pulls the same
# per-run subfolder and evaluates it. See docs/eval_workflow_hf_4090.md.
#
# Prereqs on the 4090:
#   - flower_vla_calvin cloned + venv with deps + CALVIN env installed
#   - the CALVIN task dataset present (task_ABC_D for ABC->D, task_D_D for D->D)
#   - internet access (pulls the checkpoint and Florence-2-large)
#
# Usage:
#   scripts/rtx4090/eval_calvin_from_hf.sh <repo_id> <subfolder> [num_sampling_steps] [hydra overrides...]
#
# Example (iMF, ABC->D, 1 NFE):
#   CALVIN_DATA_DIR=/data/calvin/task_ABC_D \
#   scripts/rtx4090/eval_calvin_from_hf.sh hedemil/flower-vla-calvin-ckpts imf_calvin_abcd_41936895
# ==============================================================================
set -euo pipefail

REPO_ID="${1:?usage: $0 <repo_id> <subfolder> [num_sampling_steps] [overrides...]}"
SUBFOLDER="${2:?missing <subfolder> (the per-run folder in the HF repo)}"
shift 2

# NFE: iMF = 1 (MeanFlow single step), FLOWER/RF = 4. Take an explicit positional
# arg if given, else infer from the name. Passing the wrong value silently
# mis-benchmarks the model, so it is always echoed below.
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    NFE="$1"; shift 1
elif [[ "$SUBFOLDER" == *imf* || "$REPO_ID" == *imf* ]]; then
    NFE=1
else
    NFE=4
fi

CODE_DIR="${FLOWER_CODE_DIR:-$HOME/flower_vla_calvin}"
CKPT_ROOT="${CKPT_ROOT:-$CODE_DIR/ckpts}"
CALVIN_DATA_DIR="${CALVIN_DATA_DIR:?Set CALVIN_DATA_DIR to the CALVIN task dir (e.g. /data/calvin/task_ABC_D)}"
DEVICE="${DEVICE:-0}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"

CKPT_DIR="$CKPT_ROOT/$SUBFOLDER"

echo "============================================"
echo "CALVIN eval from HF  (num_sequences=$NUM_SEQUENCES)"
echo "============================================"
echo "Repo:        $REPO_ID"
echo "Subfolder:   $SUBFOLDER"
echo "NFE:         $NFE"
echo "Data:        $CALVIN_DATA_DIR"
echo "Ckpt dir:    $CKPT_DIR"
echo "Code dir:    $CODE_DIR"
echo "============================================"

[[ -d "$CODE_DIR" ]]        || { echo "ERROR: code dir not found: $CODE_DIR (set FLOWER_CODE_DIR)"; exit 1; }
[[ -d "$CALVIN_DATA_DIR" ]] || { echo "ERROR: data dir not found: $CALVIN_DATA_DIR"; exit 1; }

# 1) Download just this run's checkpoint (model.safetensors + config.yaml).
mkdir -p "$CKPT_ROOT"
python - "$REPO_ID" "$SUBFOLDER" "$CKPT_ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo_id, subfolder, local_dir = sys.argv[1:4]
snapshot_download(repo_id=repo_id, allow_patterns=f"{subfolder}/*", local_dir=local_dir)
print("Downloaded", subfolder, "to", local_dir)
PY

[[ -f "$CKPT_DIR/model.safetensors" ]] || { echo "ERROR: model.safetensors not in $CKPT_DIR"; exit 1; }
[[ -f "$CKPT_DIR/config.yaml" ]]       || { echo "ERROR: config.yaml not in $CKPT_DIR"; exit 1; }

EVAL_OUT="$CKPT_DIR/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_OUT"

cd "$CODE_DIR"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=true

# The HF-format checkpoint dir is self-contained (model.safetensors + config.yaml),
# so train_folder and checkpoint both point at it. Single-quote the values so the
# embedded '=' in the dir name is not parsed as a Hydra override. num_sampling_steps
# is set explicitly because eval_calvin.yaml forces model.num_sampling_steps via
# eval_cfg_overwrite (default 4), which would mis-benchmark a 1-NFE iMF model.
python flower/evaluation/flower_evaluate.py \
    "train_folder='$CKPT_DIR'" \
    "checkpoint='$CKPT_DIR/model.safetensors'" \
    "dataset_path='$CALVIN_DATA_DIR'" \
    log_dir="$EVAL_OUT" \
    device="$DEVICE" \
    num_sequences="$NUM_SEQUENCES" \
    num_sampling_steps="$NFE" \
    num_videos=0 \
    log_wandb=False \
    "$@"

echo ""
echo "Done. Eval output under: $EVAL_OUT/logs/<timestamp>/"
echo "  rollout_episodes.jsonl  (per-sequence + per-subtask records)"
