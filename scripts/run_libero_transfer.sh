#!/bin/bash
# Cross-suite LIBERO transfer: evaluate a suite-trained checkpoint on a DIFFERENT
# suite. Tests whether iMF degrades more gracefully under task-distribution shift
# than RF (the generalization hypothesis).
#
# Checkpoints: checkpoints/<model>/<suite>/  (flower = RF, imf = iMF).
# Native settings by default: RF@4, iMF@1 (matches the in-suite Table R2 numbers,
# so the SR drop = in-suite - cross-suite is comparable).
#
# Usage (run inside the docker container, from the repo root):
#   bash scripts/run_libero_transfer.sh                       # MVP: libero_10 -> {spatial,object,goal}, RF+iMF
#   N_EVAL=5 SOURCES=libero_10 TARGETS=libero_spatial MODELS=imf bash scripts/run_libero_transfer.sh   # quick sanity
#   SOURCES="libero_spatial libero_object libero_goal libero_10" \
#     TARGETS="libero_spatial libero_object libero_goal libero_10" \
#     bash scripts/run_libero_transfer.sh                     # full 4x4 matrix (diagonal = in-suite baseline)
#
# Env vars: MODELS (default "flower imf"), SOURCES (default "libero_10"),
#   TARGETS (default "libero_spatial libero_object libero_goal"),
#   N_EVAL (default 20), MAX_STEPS (default 520).
#
# Output: outputs/transfer/<model>_<src>_to_<tgt>/logs/<ts>/episodes.jsonl
#   (outputs/ is mounted into the container, so results persist after `--rm` exit;
#    re-running resumes — completed cells with full episodes.jsonl are skipped.)
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"   # absolute repo root: Hydra changes cwd at runtime, so all paths must be absolute

MODELS="${MODELS:-flower imf}"
SOURCES="${SOURCES:-libero_10}"
TARGETS="${TARGETS:-libero_spatial libero_object libero_goal}"
N_EVAL="${N_EVAL:-20}"
MAX_STEPS="${MAX_STEPS:-520}"
# Write under outputs/ (mounted into the docker container, so results survive
# `docker run --rm` exit). Override with OUT=... if desired.
OUT="${OUT:-$ROOT/outputs/transfer}"
mkdir -p "$OUT"

declare -A NFE=( [flower]=4 [imf]=1 [meanflower]=1 )

fail=0
for MODEL in $MODELS; do
  steps="${NFE[$MODEL]:-1}"
  for SRC in $SOURCES; do
    CKDIR="$ROOT/checkpoints/$MODEL/$SRC"
    CKPT="$(ls "$CKDIR"/*.ckpt "$CKDIR"/model.safetensors 2>/dev/null | head -1)"
    CFG="$CKDIR/.hydra/config.yaml"
    if [[ -z "$CKPT" || ! -f "$CFG" ]]; then
      echo "SKIP $MODEL/$SRC (missing ckpt or config under $CKDIR)"; fail=1; continue
    fi
    for TGT in $TARGETS; do
      [[ "$SRC" == "$TGT" && "${INCLUDE_DIAGONAL:-0}" != "1" ]] && continue
      tag="${MODEL}_${SRC}_to_${TGT}"
      # Resumability: skip a cell whose episodes.jsonl is already complete
      # (n_eval * n_tasks lines). Re-running the same command resumes the sweep.
      n_tasks=10; [[ "$TGT" == "libero_90" ]] && n_tasks=90
      expected=$(( N_EVAL * n_tasks ))
      if [[ "${SKIP_DONE:-1}" == "1" ]]; then
        donef=""
        for f in "$OUT/$tag"/logs/*/episodes.jsonl; do
          [[ -f "$f" ]] && [[ "$(wc -l < "$f")" -ge "$expected" ]] && { donef="$f"; break; }
        done
        if [[ -n "$donef" ]]; then echo "SKIP done: $tag ($donef)"; continue; fi
      fi
      echo "=============================================="
      echo "RUN  $tag   (NFE=$steps, n_eval=$N_EVAL)"
      echo "  ckpt: $CKPT"
      echo "=============================================="
      python flower/evaluation/flower_eval_libero.py \
        train_folder="'$CFG'" \
        checkpoint="'$CKPT'" \
        benchmark_name="$TGT" \
        num_sampling_steps="$steps" \
        eval_cfg_overwrite.model.num_sampling_steps="$steps" \
        +eval_cfg_overwrite.model.load_pretrained=True \
        +variant_label="$tag" \
        log_dir="'$OUT/$tag'" \
        device=0 n_eval="$N_EVAL" max_steps="$MAX_STEPS" num_videos=0 log_wandb=false \
        || { echo "FAILED: $tag"; fail=1; }
    done
  done
done
echo ""
echo "=== transfer sweep done (fail=$fail) ==="
echo "Episodes under $OUT/*/logs/*/episodes.jsonl  (mounted -> host outputs/transfer/)"
exit $fail
