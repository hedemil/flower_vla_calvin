#!/bin/bash

# FLOWER VLA Dataset and Checkpoint Setup Script
# This script downloads and prepares CALVIN and LIBERO datasets, along with pretrained checkpoints
#
# Usage:
#   ./setup_datasets.sh [CALVIN_SPLIT] [LIBERO_BENCHMARKS] [SKIP_PREPROCESSING] [DOWNLOAD_CALVIN_CHECKPOINT] [DOWNLOAD_LIBERO_CHECKPOINT]
#
# Arguments:
#   CALVIN_SPLIT               Dataset split: D, ABC, ABCD, or debug (default: ABCD)
#   LIBERO_BENCHMARKS          LIBERO benchmarks to download:
#                              - Single: libero_goal, libero_spatial, libero_object, libero_10, libero_90
#                              - Multiple (space-separated): "libero_goal libero_spatial"
#                              - All: "all"
#                              - None/skip: "none" or empty string (default: libero_goal)
#   SKIP_PREPROCESSING         Skip CALVIN preprocessing: true or false (default: false)
#   DOWNLOAD_CALVIN_CHECKPOINT Download Calvin checkpoint: true or false (default: true)
#   DOWNLOAD_LIBERO_CHECKPOINT Download LIBERO checkpoints: true or false (default: true)
#
# Examples:
#   ./setup_datasets.sh ABCD                                    # CALVIN ABCD + libero_goal + both checkpoints
#   ./setup_datasets.sh D none                                  # Only CALVIN D + checkpoint, skip LIBERO
#   ./setup_datasets.sh ABCD "libero_goal libero_spatial"       # CALVIN + multiple LIBERO benchmarks + checkpoints
#   ./setup_datasets.sh ABCD all false false false              # All datasets, no preprocessing, no checkpoints
#   ./setup_datasets.sh ABC libero_10 false true false          # CALVIN ABC + libero_10, with CALVIN ckpt only

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if script is run from project root
if [ ! -f "scripts/download_calvin_checkpoint.sh" ]; then
    print_error "This script must be run from the project root (emil/) directory"
    exit 1
fi

echo "=========================================="
echo "FLOWER VLA Setup Script"
echo "=========================================="
echo ""

# Parse command line arguments
CALVIN_SPLIT="${1:-ABCD}"  # Default to ABCD split
LIBERO_BENCHMARKS="${2:-libero_goal}"  # Default to libero_goal
SKIP_PREPROCESSING="${3:-false}"
DOWNLOAD_CALVIN_CHECKPOINT="${4:-true}"
DOWNLOAD_LIBERO_CHECKPOINT="${5:-true}"

# Validate CALVIN split
if [[ ! "$CALVIN_SPLIT" =~ ^(D|ABC|ABCD|debug)$ ]]; then
    print_error "Invalid CALVIN split: $CALVIN_SPLIT"
    echo "Valid options: D, ABC, ABCD, debug"
    exit 1
fi

# Handle LIBERO benchmarks
SKIP_LIBERO=false
if [[ -z "$LIBERO_BENCHMARKS" ]] || [[ "$LIBERO_BENCHMARKS" == "none" ]]; then
    SKIP_LIBERO=true
    LIBERO_BENCHMARKS=""
elif [[ "$LIBERO_BENCHMARKS" == "all" ]]; then
    LIBERO_BENCHMARKS="libero_spatial libero_object libero_goal libero_90 libero_10"
fi

print_info "Configuration:"
echo "  - CALVIN Split: $CALVIN_SPLIT"
echo "  - LIBERO Benchmarks: ${LIBERO_BENCHMARKS:-none}"
echo "  - Skip Preprocessing: $SKIP_PREPROCESSING"
echo "  - Download Calvin checkpoint: $DOWNLOAD_CALVIN_CHECKPOINT"
echo "  - Download LIBERO checkpoint(s): $DOWNLOAD_LIBERO_CHECKPOINT"
echo ""

# Create necessary directories
print_info "Creating directory structure..."
mkdir -p dataset checkpoints outputs logs
print_success "Directories created"
echo ""

# ========================================
# 1. Download CALVIN Dataset
# ========================================
echo "=========================================="
echo "Step 1: Downloading CALVIN Dataset"
echo "=========================================="

CALVIN_DIR="dataset/calvin_${CALVIN_SPLIT}_D"
if [ "$CALVIN_SPLIT" = "debug" ]; then
    CALVIN_DIR="dataset/calvin_debug_dataset"
fi

if [ -d "$CALVIN_DIR" ] && [ "$(ls -A $CALVIN_DIR)" ]; then
    print_warning "CALVIN dataset ($CALVIN_SPLIT) already exists. Skipping download."
else
    print_info "Downloading CALVIN dataset: $CALVIN_SPLIT split..."
    cd dataset
    bash ../scripts/download_data.sh "$CALVIN_SPLIT"
    cd ..
    print_success "CALVIN dataset downloaded"
fi
echo ""

# ========================================
# 2. Preprocess CALVIN Dataset
# ========================================
if [ "$SKIP_PREPROCESSING" = "false" ]; then
    echo "=========================================="
    echo "Step 2: Preprocessing CALVIN Dataset"
    echo "=========================================="
    print_info "Extracting episode data to reduce I/O overhead during training..."
    print_info "This may take several minutes..."

    CALVIN_PATH="$(pwd)/dataset"
    if [ "$CALVIN_SPLIT" = "debug" ]; then
        TASK_FOLDER="calvin_debug_dataset"
    else
        TASK_FOLDER="calvin_${CALVIN_SPLIT}_D"
    fi

    # Check if already preprocessed
    if [ -f "$CALVIN_PATH/$TASK_FOLDER/training/extracted/ep_rel_actions.npy" ]; then
        print_warning "CALVIN dataset already preprocessed. Skipping."
    else
        python preprocess/extract_by_key.py \
            --in_root "$CALVIN_PATH" \
            --in_task "$TASK_FOLDER" \
            --in_split all \
            --extract_key rel_actions
        print_success "CALVIN dataset preprocessed"
    fi
    echo ""
else
    print_warning "Skipping CALVIN preprocessing (--skip-preprocessing flag set)"
    echo ""
fi

# ========================================
# 3. Download LIBERO Datasets
# ========================================
if [ "$SKIP_LIBERO" = "false" ]; then
    echo "=========================================="
    echo "Step 3: Downloading LIBERO Datasets"
    echo "=========================================="
    print_info "Downloading LIBERO benchmarks: $LIBERO_BENCHMARKS"
    print_info "Using HuggingFace mirror for reliable downloads..."

    cd LIBERO

    # Install LIBERO if needed
    if ! python -c "import libero" 2>/dev/null; then
        print_info "Installing LIBERO package..."
        pip install -e . > /dev/null 2>&1
    fi

    # Download each specified benchmark
    for benchmark in $LIBERO_BENCHMARKS; do
        if [ -d "LIBERO/$benchmark" ] && [ "$(ls -A LIBERO/$benchmark)" ]; then
            print_warning "LIBERO dataset '$benchmark' already exists. Skipping."
        else
            print_info "Downloading $benchmark..."
            python benchmark_scripts/download_libero_datasets.py --datasets "$benchmark" --use-huggingface
            print_success "$benchmark downloaded"
        fi
    done

    cd ..
    echo ""
else
    print_warning "Skipping LIBERO downloads (no benchmarks specified)"
    echo ""
fi

# ========================================
# 4. Download Calvin Checkpoint
# ========================================
echo "=========================================="
echo "Step 4: Calvin Checkpoint"
echo "=========================================="

# Skip for debug split (no checkpoint available)
if [ "$CALVIN_SPLIT" = "debug" ]; then
    print_info "Calvin checkpoint not available for debug split. Training from scratch."
    echo ""
else
    CALVIN_CKPT_DIR="checkpoints/calvin_$(echo $CALVIN_SPLIT | tr '[:upper:]' '[:lower:]')"

    if [ -d "$CALVIN_CKPT_DIR" ] && [ "$(ls -A $CALVIN_CKPT_DIR)" ]; then
        print_success "Calvin checkpoint already exists for split: $CALVIN_SPLIT"
    elif [ "$DOWNLOAD_CALVIN_CHECKPOINT" = "true" ]; then
        print_info "Downloading Calvin checkpoint for split: $CALVIN_SPLIT"
        bash scripts/download_calvin_checkpoint.sh "$CALVIN_SPLIT"
        print_success "Calvin checkpoint downloaded"
    else
        print_info "Checkpoint download skipped. To download later, run:"
        echo "  ./download_calvin_checkpoint.sh $CALVIN_SPLIT"
        print_info "You can train from scratch or download the checkpoint for evaluation."
    fi
    echo ""
fi

# ========================================
# 5. Download LIBERO Checkpoints
# ========================================
if [ "$SKIP_LIBERO" = "false" ] && [ "$DOWNLOAD_LIBERO_CHECKPOINT" = "true" ]; then
    echo "=========================================="
    echo "Step 5: LIBERO Checkpoints"
    echo "=========================================="

    # Download checkpoint for each specified benchmark
    for benchmark in $LIBERO_BENCHMARKS; do
        LIBERO_CKPT_DIR="checkpoints/${benchmark}"

        if [ -d "$LIBERO_CKPT_DIR" ] && [ "$(ls -A $LIBERO_CKPT_DIR)" ]; then
            print_success "LIBERO checkpoint already exists for: $benchmark"
        else
            print_info "Downloading LIBERO checkpoint for: $benchmark"
            if bash scripts/download_libero_checkpoint.sh "$benchmark" 2>/dev/null; then
                print_success "LIBERO checkpoint downloaded for: $benchmark"
            else
                print_warning "Checkpoint not available for $benchmark (may not exist on HuggingFace)"
                print_info "You can train from scratch or check https://huggingface.co/mbreuss"
            fi
        fi
    done
    echo ""
elif [ "$SKIP_LIBERO" = "false" ]; then
    echo "=========================================="
    echo "Step 5: LIBERO Checkpoints"
    echo "=========================================="
    print_info "LIBERO checkpoint download skipped. To download later, run:"
    echo "  ./download_libero_checkpoint.sh [benchmark_name]"
    echo ""
fi

# ========================================
# 6. Verify Setup
# ========================================
echo "=========================================="
echo "Verification Summary"
echo "=========================================="

# Check CALVIN dataset
if [ -d "$CALVIN_DIR" ] && [ "$(ls -A $CALVIN_DIR)" ]; then
    print_success "✓ CALVIN dataset ($CALVIN_SPLIT) found"

    # Check preprocessing
    if [ "$SKIP_PREPROCESSING" = "false" ]; then
        if [ -f "$CALVIN_PATH/$TASK_FOLDER/training/extracted/ep_rel_actions.npy" ]; then
            print_success "✓ CALVIN dataset preprocessed"
        else
            print_warning "✗ CALVIN dataset not preprocessed (may impact training performance)"
        fi
    fi
else
    print_error "✗ CALVIN dataset missing"
fi

# Check Calvin checkpoint (if not debug)
if [ "$CALVIN_SPLIT" != "debug" ]; then
    CALVIN_CKPT_DIR="checkpoints/calvin_$(echo $CALVIN_SPLIT | tr '[:upper:]' '[:lower:]')"
    if [ -d "$CALVIN_CKPT_DIR" ] && [ "$(ls -A $CALVIN_CKPT_DIR)" ]; then
        print_success "✓ Calvin checkpoint ($CALVIN_SPLIT) found"
    else
        print_warning "✗ Calvin checkpoint not downloaded (can train from scratch)"
    fi
fi

# Check LIBERO datasets
if [ "$SKIP_LIBERO" = "false" ]; then
    LIBERO_COUNT=0
    TOTAL_BENCHMARKS=$(echo $LIBERO_BENCHMARKS | wc -w)
    for benchmark in $LIBERO_BENCHMARKS; do
        if [ -d "LIBERO/$benchmark" ] && [ "$(ls -A LIBERO/$benchmark)" ]; then
            ((LIBERO_COUNT++))
        fi
    done
    if [ $LIBERO_COUNT -eq $TOTAL_BENCHMARKS ]; then
        print_success "✓ LIBERO datasets found ($LIBERO_COUNT/$TOTAL_BENCHMARKS benchmarks)"
    elif [ $LIBERO_COUNT -gt 0 ]; then
        print_warning "⚠ Partial LIBERO datasets found ($LIBERO_COUNT/$TOTAL_BENCHMARKS benchmarks)"
    else
        print_error "✗ LIBERO datasets missing"
    fi

    # Check LIBERO checkpoints
    if [ "$DOWNLOAD_LIBERO_CHECKPOINT" = "true" ]; then
        LIBERO_CKPT_COUNT=0
        for benchmark in $LIBERO_BENCHMARKS; do
            if [ -d "checkpoints/$benchmark" ] && [ "$(ls -A checkpoints/$benchmark)" ]; then
                ((LIBERO_CKPT_COUNT++))
            fi
        done
        if [ $LIBERO_CKPT_COUNT -eq $TOTAL_BENCHMARKS ]; then
            print_success "✓ LIBERO checkpoints found ($LIBERO_CKPT_COUNT/$TOTAL_BENCHMARKS)"
        elif [ $LIBERO_CKPT_COUNT -gt 0 ]; then
            print_warning "⚠ Partial LIBERO checkpoints ($LIBERO_CKPT_COUNT/$TOTAL_BENCHMARKS)"
        else
            print_warning "✗ LIBERO checkpoints not downloaded"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
print_info "Next steps:"
echo "  1. Build Docker image:       scripts/docker_build.sh"
echo "  2. Run Docker container:     scripts/docker_run.sh"
echo ""
print_info "Inside container - Training:"
echo "  - CALVIN:  python flower/training.py datamodule=calvin"
if [ "$SKIP_LIBERO" = "false" ]; then
    echo "  - LIBERO:  python flower/training.py datamodule=libero datamodule.datasets=[$LIBERO_BENCHMARKS]"
fi
echo ""
print_info "Inside container - Evaluation:"
echo "  - CALVIN:  ./run_evaluation.sh"
if [ "$SKIP_LIBERO" = "false" ]; then
    for benchmark in $LIBERO_BENCHMARKS; do
        echo "  - LIBERO ($benchmark):  python flower/libero_evaluation.py --benchmark $benchmark"
    done
fi
echo ""
print_info "Additional commands:"
echo "  - Download more CALVIN checkpoints:  scripts/download_calvin_checkpoint.sh [D|ABC|ABCD]"
echo "  - Download more LIBERO checkpoints:  scripts/download_libero_checkpoint.sh [benchmark_name]"
echo ""
print_info "Documentation: SETUP_GUIDE.md | QUICKSTART.md"
echo ""
