#!/bin/bash
# ==============================================================================
# LIBERO eval wrapper for the 4 iMF from-scratch ablation checkpoints.
#
# Runs flower/evaluation/flower_eval_libero.py directly with absolute paths to
# the variant's best-epoch checkpoint and its training .hydra/ dir. Tags every
# episode with +variant_label=imf_<variant> so tools/results/analyze_libero_results.py
# can group across variants.
#
# Usage:
#   bash scripts/run_libero_imf_ablation.sh <variant> <suite>
#
#   variant: default | ratio | heads | both
#   suite:   libero_10 | libero_spatial
#
# Example:
#   bash scripts/run_libero_imf_ablation.sh ratio libero_10
# ==============================================================================

set -euo pipefail

VARIANT="${1:?Usage: $0 <default|ratio|heads|both> <libero_10|libero_spatial>}"
SUITE="${2:?Usage: $0 <default|ratio|heads|both> <libero_10|libero_spatial>}"

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Best-epoch checkpoint per variant (selected by training-time eval, see results.md).
CKPT_PREFIX="/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-06"

case "$VARIANT" in
    default)
        RUN_DIR="$CKPT_PREFIX/imf_scratch_default_40945548"
        CKPT_REL="saved_models/epoch=49_eval_lh/avg_seq_len=0.90.ckpt"
        ;;
    ratio)
        RUN_DIR="$CKPT_PREFIX/imf_scratch_ratio_40945549"
        CKPT_REL="saved_models/epoch=109_eval_lh/avg_seq_len=0.93.ckpt"
        ;;
    heads)
        RUN_DIR="$CKPT_PREFIX/imf_scratch_heads_40945552"
        CKPT_REL="saved_models/epoch=39_eval_lh/avg_seq_len=0.89.ckpt"
        ;;
    both)
        RUN_DIR="$CKPT_PREFIX/imf_scratch_both_40945553"
        CKPT_REL="saved_models/epoch=109_eval_lh/avg_seq_len=0.92.ckpt"
        ;;
esac

CKPT="$RUN_DIR/$CKPT_REL"
HYDRA_CONFIG="$RUN_DIR/.hydra/config.yaml"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi
if [[ ! -f "$HYDRA_CONFIG" ]]; then
    echo "ERROR: Hydra config not found: $HYDRA_CONFIG"
    exit 1
fi

LOG_DIR="$CODE_DIR/scripts/evaluation/${SUITE}_imf_${VARIANT}_evaluation"

echo "============================================"
echo "LIBERO iMF Ablation Eval"
echo "============================================"
echo "Variant:        imf_$VARIANT"
echo "Suite:          $SUITE"
echo "Checkpoint:     $CKPT"
echo "Hydra config:   $HYDRA_CONFIG"
echo "Log dir:        $LOG_DIR"
echo "n_eval:         50"
echo "Sampling steps: 1 (iMF single-step)"
echo "============================================"

cd "$CODE_DIR"

# Quoting: the ckpt filename contains '=' chars; "'…'" passes them as literals
# rather than letting Hydra parse them as nested overrides.
python flower/evaluation/flower_eval_libero.py \
    train_folder="'$HYDRA_CONFIG'" \
    checkpoint="'$CKPT'" \
    benchmark_name="$SUITE" \
    num_sampling_steps=1 \
    eval_cfg_overwrite.model.num_sampling_steps=1 \
    +eval_cfg_overwrite.model.load_pretrained=True \
    +variant_label="imf_${VARIANT}" \
    log_dir="'$LOG_DIR'" \
    wandb_entity=VLA-Thesis \
    device=0 \
    n_eval=50 \
    max_steps=520 \
    num_videos=5 \
    log_wandb=true

echo ""
echo "============================================"
echo "Eval complete: imf_${VARIANT} on ${SUITE}"
echo "Episodes JSONL: $LOG_DIR/logs/<timestamp>/episodes.jsonl"
echo "============================================"
