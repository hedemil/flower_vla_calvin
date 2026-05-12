#!/bin/bash
# ==============================================================================
# Leonardo Setup Script (venv-based, no container)
# ==============================================================================
# Run on the Leonardo login node to prepare everything for training.
#
# Storage layout:
#   $FAST — code, checkpoints, wandb runs (fast scratch, high IOPS)
#   $WORK — Python venv, datasets, HuggingFace cache (persistent, large)
#
# Prerequisites:
#   - Set LEONARDO_FAST and LEONARDO_WORK, e.g.:
#       export LEONARDO_FAST=/leonardo_scratch/fast/<YOUR_ACCOUNT>
#       export LEONARDO_WORK=/leonardo_work/<YOUR_ACCOUNT>
#
# Usage:
#   ./scripts/leonardo/setup_leonardo.sh [step]
#
# Steps (run in order, or specify one):
#   dirs      - Create directory structure
#   venv      - Create Python venv and install dependencies
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

CODE_DIR="$FAST/project/flower_vla_calvin"
VENV_DIR="$WORK/venvs/flower_vla_calvin"
HF_CACHE="$WORK/ehed0000/hf_cache"

STEP="${1:-all}"

# ==============================================================================
# Step: dirs - Create directory structure and sync code
# ==============================================================================
setup_dirs() {
    echo "=== [1] Creating directory structure ==="

    mkdir -p "$CODE_DIR"/{logs,checkpoints,wandb_runs}
    mkdir -p "$WORK"/{venvs,hf_cache}
    mkdir -p "$WORK"/data/{calvin,libero}

    echo "Fast storage ($FAST):"
    echo "  $CODE_DIR/             (code, rsynced)"
    echo "  $CODE_DIR/logs/        (training logs)"
    echo "  $CODE_DIR/checkpoints/ (model checkpoints)"
    echo "  $CODE_DIR/wandb_runs/  (offline WandB runs)"
    echo ""
    echo "Work storage ($WORK):"
    echo "  $VENV_DIR/             (Python venv)"
    echo "  $WORK/data/calvin/     (CALVIN datasets)"
    echo "  $WORK/data/libero/     (LIBERO datasets)"
    echo "  $HF_CACHE/             (HuggingFace model cache)"
    echo ""

    # Sync entire project to fast storage (or check if already cloned)
    if [[ -f "$CODE_DIR/setup.py" ]]; then
        echo "Code already exists at $CODE_DIR"
        echo "Syncing latest changes..."
        rsync -av --delete \
            --exclude='.git' \
            --exclude='dataset' \
            --exclude='logs' \
            --exclude='checkpoints' \
            --exclude='wandb_runs' \
            "$PROJECT_ROOT/" "$CODE_DIR/"
    else
        echo "Code not found at $CODE_DIR"
        echo "Clone the repo with submodules:"
        echo "  git clone --recurse-submodules <repo_url> $CODE_DIR"
        echo "Or rsync from local:"
        echo "  rsync -av --exclude='.git' $PROJECT_ROOT/ $CODE_DIR/"
    fi

    # Ensure submodules are present
    if [[ ! -f "$CODE_DIR/calvin_env/setup.py" ]] || [[ ! -f "$CODE_DIR/LIBERO/setup.py" ]]; then
        echo ""
        echo "WARNING: Submodules (calvin_env, LIBERO) not found."
        echo "  Rsync them from your local machine:"
        echo "    rsync -av calvin_env/ leonardo:$CODE_DIR/calvin_env/"
        echo "    rsync -av LIBERO/ leonardo:$CODE_DIR/LIBERO/"
    fi

    echo "=== Directory structure ready ==="
}

# ==============================================================================
# Step: venv - Create Python venv and install all dependencies
# ==============================================================================
setup_venv() {
    echo "=== [2] Setting up Python virtual environment ==="

    module purge
    module load profile/deeplrn
    module load python/3.11.7
    module load cuda/12.1

    if [[ ! -d "$VENV_DIR" ]]; then
        echo "Creating venv at $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    else
        echo "Venv already exists at $VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"
    cd "$CODE_DIR"

    echo "Installing pip dependencies..."
    pip install --upgrade pip setuptools wheel

    # Install main requirements
    pip install -r requirements_leonardo.txt

    # Check submodules are present (rsync them if git submodule update fails)
    for subdir in calvin_env/tacto calvin_env LIBERO; do
        if [[ ! -f "$CODE_DIR/$subdir/setup.py" ]]; then
            echo "ERROR: $subdir not found at $CODE_DIR/$subdir/"
            echo "  Rsync from your local machine:"
            echo "    rsync -av $subdir/ leonardo:$CODE_DIR/$subdir/"
            exit 1
        fi
    done

    # Install submodules as editable packages
    echo "Installing tacto..."
    cd "$CODE_DIR/calvin_env/tacto"
    pip install -e .

    echo "Installing calvin_env..."
    cd "$CODE_DIR/calvin_env"
    pip install -e .

    echo "Installing LIBERO..."
    cd "$CODE_DIR/LIBERO"
    pip install --no-deps -e .

    echo "Installing flower_vla_calvin..."
    cd "$CODE_DIR"
    pip install -e .

    # Ensure correct versions (in case any submodule install overwrote them)
    pip install transformers==4.46.3 --quiet

    # Create LIBERO config file
    LIBERO_CONFIG_DIR="$HOME/.libero"
    mkdir -p "$LIBERO_CONFIG_DIR"
    cat > "$LIBERO_CONFIG_DIR/config.yaml" << EOF
benchmark_root: $CODE_DIR/LIBERO/libero/libero
bddl_files: $CODE_DIR/LIBERO/libero/libero/bddl_files
init_states: $CODE_DIR/LIBERO/libero/libero/init_files
datasets: $WORK/data/libero
assets: $CODE_DIR/LIBERO/libero/libero/assets
EOF
    echo "LIBERO config written to $LIBERO_CONFIG_DIR/config.yaml"

    echo ""
    echo "=== Venv ready ==="
}

# ==============================================================================
# Step: calvin - Download and preprocess CALVIN
# ==============================================================================
setup_calvin() {
    # Which CALVIN task to download. Defaults to task_D_D (single env,
    # smaller, ~80 GB). For cross-env, set CALVIN_TASK=task_ABC_D (~700 GB).
    local TASK="${CALVIN_TASK:-task_D_D}"
    echo "=== [3] Downloading CALVIN dataset ($TASK) ==="

    CALVIN_DIR="$WORK/data/calvin"
    mkdir -p "$CALVIN_DIR"
    cd "$CALVIN_DIR"

    if [[ -d "$CALVIN_DIR/$TASK" ]]; then
        echo "$TASK already exists, skipping download"
    else
        echo "Downloading $TASK..."
        wget -q --show-progress "http://calvin.cs.uni-freiburg.de/dataset/${TASK}.zip"
        unzip -q "${TASK}.zip"
        rm "${TASK}.zip"
        echo "Downloaded $TASK to $CALVIN_DIR/$TASK"
    fi

    # Preprocess: extract rel_actions
    echo ""
    echo "Preprocessing CALVIN data (extracting rel_actions)..."
    source "$VENV_DIR/bin/activate"
    python "$CODE_DIR/preprocess/extract_by_key.py" \
        -i "$CALVIN_DIR" \
        --in_task "$TASK" \
        --in_split all \
        -k rel_actions

    echo ""
    echo "=== CALVIN $TASK setup complete ==="
}

# ==============================================================================
# Step: libero - Download LIBERO datasets
# ==============================================================================
setup_libero() {
    echo "=== [4] Downloading LIBERO datasets ==="

    source "$VENV_DIR/bin/activate"

    echo "Downloading libero_goal libero_spatial libero_object..."
    cd "$CODE_DIR/LIBERO"
    python benchmark_scripts/download_libero_datasets.py \
        --datasets libero_goal --use-huggingface
    python benchmark_scripts/download_libero_datasets.py \
        --datasets libero_spatial --use-huggingface
    python benchmark_scripts/download_libero_datasets.py \
        --datasets libero_object --use-huggingface

    # Download libero_10 and libero_90 directly from HuggingFace.
    # The official download script only supports libero_100 (which doesn't exist
    # on HF), but the HF repo has libero_10 and libero_90 as separate folders.
    LIBERO_DATA_DIR="$WORK/data/libero"
    echo "Downloading libero_10 and libero_90 from HuggingFace..."
    python -c "
from huggingface_hub import snapshot_download
for ds in ['libero_10', 'libero_90']:
    print(f'Downloading {ds}...')
    snapshot_download(
        repo_id='yifengzhu-hf/LIBERO-datasets',
        repo_type='dataset',
        local_dir='$LIBERO_DATA_DIR',
        allow_patterns=f'{ds}/*',
        local_dir_use_symlinks=False,
    )
    print(f'{ds} done')
"

    echo ""
    echo "=== LIBERO setup complete ==="
}

# ==============================================================================
# Step: hf_cache - Pre-cache HuggingFace models
# ==============================================================================
setup_hf_cache() {
    echo "=== [5] Pre-caching HuggingFace models ==="

    source "$VENV_DIR/bin/activate"
    export HF_HOME="$HF_CACHE"
    python "$CODE_DIR/scripts/leonardo/download_hf_models.py"

    echo ""
    echo "=== HuggingFace cache ready at $HF_CACHE/ ==="
}

# ==============================================================================
# Main
# ==============================================================================
case "$STEP" in
    dirs)
        setup_dirs
        ;;
    venv)
        setup_venv
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
        setup_venv
        echo ""
        setup_calvin
        echo ""
        setup_libero
        echo ""
        setup_hf_cache
        echo ""
        echo "=== All setup complete ==="
        echo ""
        echo "Next steps:"
        echo "  1. Set your SLURM account:"
        echo "     sed -i 's/<YOUR_ACCOUNT>/your_account/g' $CODE_DIR/scripts/leonardo/sbatch_*.sh"
        echo "  2. Run a debug job:"
        echo "     sbatch $CODE_DIR/scripts/leonardo/sbatch_debug.sh libero"
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Usage: $0 [dirs|venv|calvin|libero|hf_cache|all]"
        exit 1
        ;;
esac
