#!/bin/bash

# Script to download CALVIN-finetuned FLOWER checkpoint from Hugging Face

set -e

echo "=========================================="
echo "Downloading CALVIN-finetuned Checkpoint"
echo "=========================================="

# Choose which checkpoint to download
SPLIT=${1:-D}

case $SPLIT in
    D)
        echo "Downloading CALVIN D split checkpoint..."
        REPO="mbreuss/flower_calvin_d"
        ;;
    ABC)
        echo "Downloading CALVIN ABC split checkpoint..."
        REPO="mbreuss/flower_calvin_abc"
        ;;
    ABCD)
        echo "Downloading CALVIN ABCD split checkpoint..."
        REPO="mbreuss/flower_calvin_abcd"
        ;;
    *)
        echo "Invalid split: $SPLIT"
        echo "Usage: $0 [D|ABC|ABCD]"
        exit 1
        ;;
esac

CHECKPOINT_DIR="checkpoints/calvin_${SPLIT,,}"
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
