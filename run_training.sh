#!/bin/bash

# Script to run FLOWER evaluation on CALVIN debug dataset

set -e

echo "=========================================="
echo "Running FLOWER Evaluation on CALVIN"
echo "=========================================="

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if datasets exist
if [ ! -d "${SCRIPT_DIR}/dataset/calvin_debug_dataset" ]; then
    echo "ERROR: CALVIN debug dataset not found!"
    echo "Please run: cd dataset && sh download_data.sh debug"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Dataset: ${SCRIPT_DIR}/dataset/calvin_debug_dataset"
echo "  Device: cuda:0"
echo ""

# Run training with absolute paths (Hydra changes CWD, so relative paths won't work)
python flower/training_calvin.py \
    root_data_dir=${SCRIPT_DIR}/dataset/calvin_debug_dataset \
    lang_folder=lang_annotations \
    log_dir=${SCRIPT_DIR}/training/calvin_debug_training \
    max_epochs=1 \
    devices=1 \
    logger.entity=VLA-Thesis \
    logger.project=calvin_d \
    logger.group=debug \
    logger.name=run1 \
    device=0 \