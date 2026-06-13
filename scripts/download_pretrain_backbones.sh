#!/bin/bash

# Download the from-scratch flowereef pretrain backbones (RF + iMF) from HuggingFace.
# Companion to scripts/leonardo/upload_pretrain_hf.py.

set -e

REPO="${1:-hedemil/flower-vla-flowereef-pretrain}"
DEST="${2:-checkpoints/pretrained_flowereef}"

echo "=========================================="
echo "Downloading flowereef pretrain backbones"
echo "  repo: $REPO"
echo "  dest: $DEST"
echo "=========================================="

mkdir -p "$DEST"

# huggingface-cli ships with the huggingface_hub package (already a project dep).
# Pulls both subfolders (rf/, imf/), each with model.safetensors + config.yaml.
huggingface-cli download "$REPO" \
    --repo-type model \
    --local-dir "$DEST"

echo ""
echo "=========================================="
echo "Download complete. Layout:"
echo "=========================================="
find "$DEST" -maxdepth 2 -type f | sort
