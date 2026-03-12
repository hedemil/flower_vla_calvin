#!/bin/bash

# Script to run FLOWER training on LIBERO datasets
# All Hydra overrides are passed directly through to the training script.
#
# Usage:
#   ./scripts/run_training_libero_cluster.sh [HYDRA_OVERRIDES...]
#
# Examples:
#   # Standard FLOWERVLA on libero_spatial
#   ./scripts/run_training_libero_cluster.sh libero_benchmark=libero_spatial
#
#   # Decoupled MeanFlow from pretrained checkpoint
#   ./scripts/run_training_libero_cluster.sh \
#       libero_benchmark=libero_spatial \
#       model=decoupled_meanflower \
#       model.load_pretrained=True \
#       model.pretrained_model_path=checkpoints/pretrained/360000_model_weights.pt
#
#   # Override batch size, epochs, etc.
#   ./scripts/run_training_libero_cluster.sh \
#       libero_benchmark=libero_goal \
#       model=meanflower \
#       batch_size=8 \
#       max_epochs=30

set -e

echo "=========================================="
echo "Running FLOWER LIBERO Training"
echo "=========================================="

# Get project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Set CUDA memory allocator config for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "Overrides: $@"
echo ""

# Verify GPUs
echo "Detected GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPUs detected)"
echo ""

# Run training — all arguments are forwarded as Hydra overrides
python ${PROJECT_ROOT}/flower/training_libero.py "$@"
