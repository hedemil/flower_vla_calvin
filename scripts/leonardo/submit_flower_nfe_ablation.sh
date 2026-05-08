#!/bin/bash
# ==============================================================================
# Submit flower NFE x benchmark ablation: 2 benchmarks x 3 NFE = 6 jobs.
# Uses the pretrained flower checkpoint (auto-set in sbatch_train_libero.sh).
#
# Matrix:
#   benchmarks: libero_10, libero_spatial
#   NFE:        1, 2, 3   (model.num_sampling_steps)
#
# Usage:
#   bash scripts/leonardo/submit_flower_nfe_ablation.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/sbatch_train_libero.sh"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "ERROR: sbatch script not found: $SBATCH_SCRIPT"
    exit 1
fi

for BM in libero_10 libero_spatial; do
    for N in 1 2 3; do
        echo "Submitting: benchmark=$BM nfe=$N"
        sbatch \
            --export=ALL,DATASET="$BM",RUN_TAG="nfe${N}" \
            -J "flower-${BM}-nfe${N}" \
            "$SBATCH_SCRIPT" flower \
            "model.num_sampling_steps=${N}"
    done
done
