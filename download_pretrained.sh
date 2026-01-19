#!/bin/bash

# Script to download pretrained FLOWER checkpoint from Hugging Face

set -e

echo "=========================================="
echo "Downloading Pretrained FLOWER Checkpoint"
echo "=========================================="

CHECKPOINT_DIR="checkpoints/pretrained"
mkdir -p $CHECKPOINT_DIR

cd $CHECKPOINT_DIR

echo ""
echo "Downloading model weights (1.67 GB)..."
wget https://huggingface.co/mbreuss/flower_vla_pret/resolve/main/360000_model_weights.pt

echo ""
echo "Downloading config file..."
wget https://huggingface.co/mbreuss/flower_vla_pret/resolve/main/config.yaml

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo "Files saved in: $(pwd)"
echo ""
echo "Checkpoint: 360000_model_weights.pt"
echo "Config: config.yaml"
