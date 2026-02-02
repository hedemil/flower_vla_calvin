#!/bin/bash

#SBATCH --job-name=FLOWER_A6000
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mem=128G

# Define output and error files
#SBATCH --output=logs/slurm_outputs/%x_%j.out
#SBATCH --error=logs/slurm_outputs/%x_%j.err

set -e

echo "=========================================="
echo "Running FLOWER Training on Cluster (SLURM)"
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "=========================================="

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create log directory if it doesn't exist
mkdir -p ${SCRIPT_DIR}/logs/slurm_outputs

# Check if datasets exist
if [ ! -d "${SCRIPT_DIR}/dataset/calvin_debug_dataset" ]; then
    echo "ERROR: CALVIN debug dataset not found!"
    echo "Please run: ./download_data.sh debug"
    exit 1
fi

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate flower_cal

# Verify GPUs
echo ""
echo "Detected GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Create timestamp for unique run identification
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Configuration:"
echo "  Dataset: ${SCRIPT_DIR}/dataset/calvin_debug_dataset"
echo "  GPUs: 2 (SLURM allocated)"
echo "  Batch size: 4 (2 per GPU with FSDP)"
echo "  Max epochs: 100"
echo "  Multi-GPU: FSDP strategy (memory optimized)"
echo "  Camera views: Single (static only)"
echo "  EMA: Delayed start (step 5000)"
echo ""

# Set CUDA memory allocator config for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training with SLURM and memory-optimized settings
# FSDP strategy shards model/optimizer across GPUs to reduce per-GPU memory
# EMA delayed to avoid 4GB extra memory during early training
srun python ${SCRIPT_DIR}/flower/training_calvin.py \
    root_data_dir=${SCRIPT_DIR}/dataset/calvin_debug_dataset \
    lang_folder=lang_annotations \
    log_dir=${SCRIPT_DIR}/logs/training_${TIMESTAMP} \
    batch_size=4 \
    max_epochs=100 \
    devices=2 \
    logger.entity=VLA-Thesis \
    logger.project=calvin_a6000 \
    logger.group=cluster_training \
    logger.name=a6000_2gpu_fsdp_${TIMESTAMP} \
    model.freeze_florence=False \
    model.freeze_vision_tower=False \
    model.use_second_view=False \
    trainer.strategy=fsdp \
    trainer.num_sanity_val_steps=2 \
    callbacks.ema.start_step=5000

echo ""
echo "Training completed!"
echo "Logs saved to: ${SCRIPT_DIR}/logs/training_${TIMESTAMP}"
