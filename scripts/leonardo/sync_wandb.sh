#!/bin/bash
# ==============================================================================
# Sync offline WandB runs from Leonardo (run on login node)
#
# Usage:
#   ./scripts/leonardo/sync_wandb.sh
# ==============================================================================

set -euo pipefail

FAST="${LEONARDO_FAST:-${FAST:?Set LEONARDO_FAST or FAST}}"
LOGS_DIR="$FAST/flower_vla_calvin/logs/runs"

if [[ ! -d "$LOGS_DIR" ]]; then
    echo "Logs directory not found: $LOGS_DIR"
    exit 1
fi

# Find all offline runs (nested inside Hydra run dirs under wandb/)
OFFLINE_RUNS=$(find "$LOGS_DIR" -type d -name "offline-run-*" 2>/dev/null)

if [[ -z "$OFFLINE_RUNS" ]]; then
    echo "No offline WandB runs found in $LOGS_DIR"
    exit 0
fi

COUNT=$(echo "$OFFLINE_RUNS" | wc -l)
echo "Found $COUNT offline WandB run(s) to sync"
echo ""

SYNCED=0
FAILED=0

for run_dir in $OFFLINE_RUNS; do
    echo "Syncing: $(basename "$run_dir")"
    if wandb sync "$run_dir" 2>&1; then
        echo "  Synced successfully"
        ((SYNCED++))
    else
        echo "  FAILED to sync"
        ((FAILED++))
    fi
    echo ""
done

echo "============================================"
echo "Synced: $SYNCED / $COUNT"
if [[ $FAILED -gt 0 ]]; then
    echo "Failed: $FAILED"
fi
echo "============================================"
