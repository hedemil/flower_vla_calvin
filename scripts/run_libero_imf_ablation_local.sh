#!/bin/bash
# ==============================================================================
# Local LIBERO eval wrapper for the 4 iMF from-scratch ablation checkpoints.
#
# Reads checkpoints from the local layout produced by the rsync block in
# docs/download_checkpoints.md (under checkpoints/imf_ablation/libero_10_<variant>/),
# and evaluates against any LIBERO suite (decoupled from the checkpoint dir name,
# so the same LIBERO-10-trained ckpt can be evaluated on libero_spatial too).
#
# Tags every episode with +variant_label=imf_<variant> so
# tools/results/analyze_libero_results.py can group across variants.
#
# Usage:
#   bash scripts/run_libero_imf_ablation_local.sh <variant> <suite>
#
#   variant: default | ratio | heads | both
#   suite:   libero_10 | libero_spatial
#
# Example:
#   bash scripts/run_libero_imf_ablation_local.sh ratio libero_10
# ==============================================================================

set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-disabled}"

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

CKPT_DIR="$CODE_DIR/checkpoints/imf_ablation/libero_10_${VARIANT}"
HYDRA_CONFIG="$CKPT_DIR/.hydra/config.yaml"

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "ERROR: Checkpoint dir not found: $CKPT_DIR"
    echo "  Run the rsync commands in docs/download_checkpoints.md (iMF from-scratch ablation block) first."
    exit 1
fi
if [[ ! -f "$HYDRA_CONFIG" ]]; then
    echo "ERROR: Hydra config not found: $HYDRA_CONFIG"
    exit 1
fi

# Auto-detect the .ckpt file (filename embeds avg_seq_len, varies per variant).
CKPT=$(find "$CKPT_DIR" -maxdepth 1 -name "*.ckpt" -type f | head -1)
if [[ -z "$CKPT" ]]; then
    echo "ERROR: No .ckpt file found at top level of $CKPT_DIR"
    exit 1
fi

LOG_DIR="$CODE_DIR/scripts/evaluation/${SUITE}_imf_${VARIANT}_evaluation"

echo "============================================"
echo "LIBERO iMF Ablation Eval (local)"
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

# Quoting: ckpt filenames contain '=' (e.g. avg_seq_len=0.93.ckpt); "'…'" passes
# them as literals so Hydra doesn't parse them as nested overrides.
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
