#!/bin/bash
# ==============================================================================
# One-time setup for Leonardo HPC (CINECA)
# Run on login node (has internet access)
#
# Prerequisites:
#   - LEONARDO_FAST and LEONARDO_WORK env vars set (typically $FAST and $WORK)
#   - Singularity container already transferred to $LEONARDO_WORK/containers/
#
# Usage:
#   ./scripts/leonardo/setup_leonardo.sh [step]
#
# Steps (run in order, or specify one):
#   dirs      - Create directory structure
#   calvin    - Download + preprocess CALVIN dataset
#   libero    - Download LIBERO datasets
#   hf_cache  - Pre-cache HuggingFace models (Florence-2)
#   all       - Run all steps (default)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------
# Validate environment
# ---------------------
FAST="${LEONARDO_FAST:-${FAST:-}}"
WORK="${LEONARDO_WORK:-${WORK:-}}"

if [[ -z "$FAST" ]]; then
    echo "ERROR: Set LEONARDO_FAST (or FAST) to your fast-scratch path"
    echo "  e.g., export LEONARDO_FAST=\$CINECA_SCRATCH"
    exit 1
fi
if [[ -z "$WORK" ]]; then
    echo "ERROR: Set LEONARDO_WORK (or WORK) to your persistent work path"
    echo "  e.g., export LEONARDO_WORK=\$CINECA_WORK"
    exit 1
fi

PROJECT_FAST="$FAST/flower_vla_calvin"
SIF="$WORK/containers/flower_vla_calvin.sif"

STEP="${1:-all}"

# Helper to run a command inside the Singularity container (CPU-only, login node)
run_in_container() {
    singularity exec \
        --no-home \
        --env HOME=/appuser \
        --bind "$WORK/data/calvin:/workspace/flower_vla_calvin/dataset" \
        --bind "$WORK/data/libero:/workspace/flower_vla_calvin/LIBERO/libero/datasets" \
        --bind "$WORK/hf_cache:/appuser/.cache/huggingface" \
        --bind "$PROJECT_FAST/conf:/workspace/flower_vla_calvin/conf" \
        --bind "$PROJECT_FAST/flower:/workspace/flower_vla_calvin/flower" \
        "$SIF" \
        "$@"
}

# ==============================================================================
# Step: dirs - Create directory structure
# ==============================================================================
setup_dirs() {
    echo "=== Creating directory structure ==="

    mkdir -p "$PROJECT_FAST"/{logs,checkpoints,wandb_runs,conf,flower}
    mkdir -p "$WORK"/{containers,hf_cache}
    mkdir -p "$WORK"/data/{calvin,libero}

    echo "Fast storage ($FAST):"
    echo "  $PROJECT_FAST/logs/"
    echo "  $PROJECT_FAST/checkpoints/"
    echo "  $PROJECT_FAST/wandb_runs/"
    echo "  $PROJECT_FAST/conf/        (bind-mount for config iteration)"
    echo "  $PROJECT_FAST/flower/      (bind-mount for code iteration)"
    echo ""
    echo "Work storage ($WORK):"
    echo "  $WORK/containers/          (Singularity .sif image)"
    echo "  $WORK/data/calvin/         (CALVIN datasets)"
    echo "  $WORK/data/libero/         (LIBERO datasets)"
    echo "  $WORK/hf_cache/            (HuggingFace model cache)"
    echo ""

    # Sync code for bind-mount iteration
    echo "Syncing conf/ and flower/ to fast storage for bind-mount iteration..."
    rsync -av --delete "$PROJECT_ROOT/conf/" "$PROJECT_FAST/conf/"
    rsync -av --delete "$PROJECT_ROOT/flower/" "$PROJECT_FAST/flower/"

    echo ""
    echo "=== Directory structure ready ==="
    echo ""
    echo "Container setup (choose one):"
    echo "  Option A - Pull from registry:"
    echo "    singularity pull --dir $WORK/containers/ docker://ghcr.io/<USER>/flower_vla_calvin:latest"
    echo "    mv $WORK/containers/flower_vla_calvin_latest.sif $SIF"
    echo ""
    echo "  Option B - Build locally, then rsync:"
    echo "    # On your local machine:"
    echo "    singularity build flower_vla_calvin.sif docker-daemon://flower_vla_calvin:latest"
    echo "    rsync -avP flower_vla_calvin.sif leonardo:$WORK/containers/"
    echo ""
}

# ==============================================================================
# Step: calvin - Download and preprocess CALVIN
# ==============================================================================
setup_calvin() {
    echo "=== Downloading CALVIN dataset ==="

    CALVIN_DIR="$WORK/data/calvin"
    cd "$CALVIN_DIR"

    # Download task_D_D (single environment, ~7GB)
    if [[ -d "$CALVIN_DIR/task_D_D" ]]; then
        echo "task_D_D already exists, skipping download"
    else
        echo "Downloading task_D_D..."
        wget -q --show-progress http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
        unzip -q task_D_D.zip
        rm task_D_D.zip
        echo "Downloaded task_D_D to $CALVIN_DIR/task_D_D"
    fi

    # Check container exists for preprocessing
    if [[ ! -f "$SIF" ]]; then
        echo ""
        echo "WARNING: Container not found at $SIF"
        echo "Cannot run preprocessing. Transfer the container first, then re-run:"
        echo "  $0 calvin"
        return 1
    fi

    # Preprocess: extract rel_actions (required for use_extracted_rel_actions=true)
    echo ""
    echo "Preprocessing CALVIN data (extracting rel_actions)..."
    run_in_container python /workspace/flower_vla_calvin/preprocess/extract_by_key.py \
        -i /workspace/flower_vla_calvin/dataset \
        --in_task task_D_D \
        --in_split all \
        -k rel_actions

    echo ""
    echo "=== CALVIN setup complete ==="
}

# ==============================================================================
# Step: libero - Download LIBERO datasets
# ==============================================================================
setup_libero() {
    echo "=== Downloading LIBERO datasets ==="

    if [[ ! -f "$SIF" ]]; then
        echo "ERROR: Container not found at $SIF"
        echo "LIBERO download requires the container (needs HuggingFace libraries)"
        return 1
    fi

    # Download libero_spatial (default benchmark)
    echo "Downloading libero_spatial..."
    run_in_container bash -c "
        cd /workspace/flower_vla_calvin/LIBERO && \
        python benchmark_scripts/download_libero_datasets.py \
            --datasets libero_spatial --use-huggingface
    "

    echo ""
    echo "To download additional benchmarks, run inside container:"
    echo "  libero_goal, libero_object, libero_10, libero_90"
    echo ""
    echo "=== LIBERO setup complete ==="
}

# ==============================================================================
# Step: hf_cache - Pre-cache HuggingFace models
# ==============================================================================
setup_hf_cache() {
    echo "=== Pre-caching HuggingFace models ==="

    if [[ ! -f "$SIF" ]]; then
        echo "ERROR: Container not found at $SIF"
        return 1
    fi

    run_in_container python /workspace/flower_vla_calvin/scripts/leonardo/download_hf_models.py

    echo ""
    echo "=== HuggingFace cache ready at $WORK/hf_cache/ ==="
}

# ==============================================================================
# Main
# ==============================================================================
case "$STEP" in
    dirs)
        setup_dirs
        ;;
    calvin)
        setup_calvin
        ;;
    libero)
        setup_libero
        ;;
    hf_cache)
        setup_hf_cache
        ;;
    all)
        setup_dirs
        echo ""
        if [[ -f "$SIF" ]]; then
            setup_calvin
            echo ""
            setup_libero
            echo ""
            setup_hf_cache
        else
            echo "Container not found at $SIF"
            echo "Transfer the container first, then re-run to complete data setup:"
            echo "  $0 calvin"
            echo "  $0 libero"
            echo "  $0 hf_cache"
        fi
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Usage: $0 [dirs|calvin|libero|hf_cache|all]"
        exit 1
        ;;
esac
