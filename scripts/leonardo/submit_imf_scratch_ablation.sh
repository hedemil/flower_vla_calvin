#!/bin/bash
# ==============================================================================
# Submit all four iMF from-scratch LIBERO-10 ablation runs in one go.
#
# Variants: default | ratio | heads | both
# (See sbatch_train_libero_imf_scratch.sh for the parameter matrix.)
#
# Usage:
#   bash scripts/leonardo/submit_imf_scratch_ablation.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/sbatch_train_libero_imf_scratch.sh"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "ERROR: sbatch script not found: $SBATCH_SCRIPT"
    exit 1
fi

for v in default ratio heads both; do
    echo "Submitting variant: $v"
    sbatch -J "imf-libero-scratch-${v}" "$SBATCH_SCRIPT" "$v"
done
