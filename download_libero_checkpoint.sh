#!/bin/bash

# Script to download LIBERO-finetuned FLOWER checkpoint from Hugging Face

set -e

echo "=========================================="
echo "Downloading LIBERO-finetuned Checkpoint"
echo "=========================================="

# Choose which checkpoint to download
BENCHMARK=${1:-libero_goal}

case $BENCHMARK in
    libero_spatial)
        echo "Downloading LIBERO Spatial checkpoint..."
        REPO="mbreuss/flower_libero_spatial"
        ;;
    libero_object)
        echo "Downloading LIBERO Object checkpoint..."
        REPO="mbreuss/flower_libero_object"
        ;;
    libero_goal)
        echo "Downloading LIBERO Goal checkpoint..."
        REPO="mbreuss/flower_libero_goal"
        ;;
    libero_10)
        echo "Downloading LIBERO 10 checkpoint..."
        REPO="mbreuss/flower_libero_10"
        ;;
    libero_90)
        echo "Downloading LIBERO 90 checkpoint..."
        REPO="mbreuss/flower_libero_90"
        ;;
    *)
        echo "Invalid benchmark: $BENCHMARK"
        echo "Usage: $0 [libero_spatial|libero_object|libero_goal|libero_10|libero_90]"
        exit 1
        ;;
esac

CHECKPOINT_DIR="checkpoints/${BENCHMARK}"
mkdir -p $CHECKPOINT_DIR

cd $CHECKPOINT_DIR

echo ""
echo "Downloading from HuggingFace: $REPO"
echo ""

# Clone the repository using git (HuggingFace uses git-lfs)
git clone https://huggingface.co/$REPO .
rm -rf .git  # Remove git metadata to save space

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo "Files saved in: $(pwd)"
echo ""
ls -lh
