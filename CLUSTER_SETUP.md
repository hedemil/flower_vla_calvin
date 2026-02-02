# Cluster Training Setup Guide

Guide for running FLOWER VLA training on cluster with 2x NVIDIA A6000 GPUs.

---

## Prerequisites

- VPN credentials for remote access
- SSH credentials for the cluster
- Git configured with SSH keys or personal access token
- WandB account and API key

---

## Step 1: Connect to VPN

```bash
# Replace with your VPN connection command
# Example for OpenVPN:
sudo openvpn --config /path/to/your/vpn-config.ovpn

# Example for Cisco AnyConnect:
# /opt/cisco/anyconnect/bin/vpn connect YOUR_VPN_SERVER
# Enter username and password when prompted

# Verify VPN connection
ping <CLUSTER_IP>  # Replace with actual cluster IP
```

**Note:** Keep the VPN connection active throughout your training session.

---

## Step 2: SSH into the Cluster

```bash
# SSH to cluster
ssh your_username@cluster.address.com

# If using a specific port:
# ssh -p 2222 your_username@cluster.address.com

# Verify GPU availability
nvidia-smi

# Expected output: 2x NVIDIA A6000 (48GB each)
```

---

## Step 3: Clone the Repository

```bash
# Navigate to your workspace
cd /path/to/your/workspace  # e.g., ~/projects or /scratch/username

# Clone the repository
git clone git@github.com:your-username/VLA-master-thesis-2026.git
# OR if using HTTPS:
# git clone https://github.com/your-username/VLA-master-thesis-2026.git

# Navigate to project directory
cd VLA-master-thesis-2026/emil

# Verify you're on the correct branch
git status
git checkout scripts/train  # or your desired branch
```

---

## Step 4: Build Docker Image

```bash
# Make sure you're in the emil directory
cd /path/to/VLA-master-thesis-2026/emil

# Build the Docker image (this will take 15-30 minutes)
docker build -t flower_vla_calvin:latest .

# Verify the image was built successfully
docker images | grep flower_vla_calvin
```

**Troubleshooting:**
- If build fails due to disk space, clean up old images: `docker system prune -a`
- If build fails on dependencies, check your internet connection

---

## Step 5: Prepare Data and Configuration

### 5.1 Download CALVIN Dataset

```bash
# Inside the emil directory
# Download the CALVIN dataset (choose one):

# For debug dataset (small, ~1GB):
./download_data.sh debug

# For full ABC→D dataset (~50GB):
# ./download_data.sh ABC

# For full ABCD dataset:
# ./download_data.sh ABCD
```

### 5.2 Configure WandB

```bash
# Set up WandB authentication
export WANDB_API_KEY="your_wandb_api_key_here"

# Or login interactively (will be prompted when training starts):
# wandb login
```

### 5.3 Update Training Configuration

Before running, verify/update the configuration:

```bash
# Check current config
cat conf/config_calvin.yaml

# Key parameters for A6000 training:
# - batch_size: 64 (or 32 if using 2 GPUs with DDP)
# - devices: 2 (to use both A6000s)
# - max_epochs: 100 (or your desired number)
# - freeze_florence: False (unfreeze for full training)
# - freeze_vision_tower: False
# - use_second_view: True (use both camera views)
```

**Optional:** Create a custom training script for cluster:

```bash
cat > run_training_cluster.sh << 'EOF'
#!/bin/bash

set -e

echo "=========================================="
echo "Running FLOWER Training on Cluster"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Verify datasets
if [ ! -d "${SCRIPT_DIR}/dataset/calvin_debug_dataset" ]; then
    echo "ERROR: CALVIN dataset not found!"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Dataset: ${SCRIPT_DIR}/dataset/calvin_debug_dataset"
echo "  GPUs: 2x NVIDIA A6000"
echo "  Batch size: 64"
echo ""

# Run training with cluster-optimized settings
python flower/training_calvin.py \
    root_data_dir=${SCRIPT_DIR}/dataset/calvin_debug_dataset \
    lang_folder=lang_annotations \
    log_dir=${SCRIPT_DIR}/logs/training_$(date +%Y%m%d_%H%M%S) \
    batch_size=64 \
    max_epochs=100 \
    devices=2 \
    logger.entity=VLA-Thesis \
    logger.project=calvin_a6000 \
    logger.group=cluster_training \
    logger.name=a6000_2gpu_$(date +%Y%m%d_%H%M%S) \
    model.freeze_florence=False \
    model.freeze_vision_tower=False \
    model.use_second_view=True
EOF

chmod +x run_training_cluster.sh
```

---

## Step 6: Run Training

### 6.1 Start Docker Container

```bash
# Make docker_run.sh executable
chmod +x docker_run.sh

# Launch container
./docker_run.sh
```

This will:
- Mount all necessary directories
- Enable GPU access
- Set up environment variables
- Start an interactive bash shell inside the container

### 6.2 Inside the Docker Container

```bash
# Verify GPUs are accessible
nvidia-smi

# Expected: 2x NVIDIA A6000, ~48GB each

# Run training
./run_training_cluster.sh

# OR use the original script with custom parameters:
# ./run_training.sh
```

### 6.3 Monitor Training

**In a new terminal (outside Docker):**

```bash
# SSH to cluster again
ssh your_username@cluster.address.com

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
cd /path/to/VLA-master-thesis-2026/emil
tail -f logs/training_*/training.log  # Adjust path as needed
```

**Via WandB:**
- Navigate to https://wandb.ai/VLA-Thesis/calvin_a6000
- Monitor metrics in real-time

### 6.4 Training with tmux/screen (Recommended)

To keep training running if SSH disconnects:

```bash
# Start tmux session
tmux new -s flower_training

# Inside tmux:
./docker_run.sh
# Then run training

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t flower_training
```

---

## Step 7: Monitor and Manage Training

### Check Training Progress

```bash
# Inside Docker container:
# Training will create checkpoints in:
ls -lh saved_models/

# Check WandB logs
# Training metrics are automatically uploaded to WandB
```

### Stop Training Gracefully

```bash
# Inside Docker container, press:
Ctrl+C

# This will:
# - Save current checkpoint
# - Upload final metrics to WandB
# - Exit gracefully
```

### Resume Training (if interrupted)

```bash
# Modify run_training_cluster.sh to add checkpoint path:
# Add parameter: ckpt_path=/path/to/checkpoint.ckpt

# Or run directly:
python flower/training_calvin.py \
    ... (your previous parameters) \
    ckpt_path=saved_models/your_checkpoint.ckpt
```

---

## Step 8: Clean Up After Training

### 8.1 Copy Important Files

```bash
# Exit Docker container
exit

# On the cluster, copy results to permanent storage
cd /path/to/VLA-master-thesis-2026/emil

# Copy model checkpoints
mkdir -p ~/trained_models/flower_$(date +%Y%m%d)
cp saved_models/*.ckpt ~/trained_models/flower_$(date +%Y%m%d)/

# Copy logs
cp -r logs ~/trained_models/flower_$(date +%Y%m%d)/

# Copy configuration
cp conf/config_calvin.yaml ~/trained_models/flower_$(date +%Y%m%d)/
cp conf/model/flower.yaml ~/trained_models/flower_$(date +%Y%m%d)/
```

### 8.2 Download Results Locally (Optional)

```bash
# On your local machine:
# Download checkpoints
scp -r your_username@cluster.address.com:~/trained_models/flower_YYYYMMDD ./

# Or use rsync for large files:
rsync -avz --progress your_username@cluster.address.com:~/trained_models/flower_YYYYMMDD ./
```

### 8.3 Clean Up Cluster Storage

```bash
# On the cluster:

# Remove Docker container (if still running)
docker ps  # List running containers
docker stop <container_id>  # If needed

# Remove dataset (if you don't need it anymore)
cd /path/to/VLA-master-thesis-2026/emil
rm -rf dataset/calvin_debug_dataset  # Or full dataset

# Clean up Docker images (if not needed)
docker system prune -a

# Remove training logs (after copying important ones)
rm -rf logs/training_*

# Remove cached files
rm -rf ~/.cache/huggingface/hub/*  # HuggingFace model cache
rm -rf ~/.cache/wandb/*  # WandB cache

# IMPORTANT: Keep the trained model checkpoints safe!
# Only delete after confirming they're backed up
```

### 8.4 Verify Disk Usage

```bash
# Check disk usage
df -h

# Check directory sizes
du -sh ~/trained_models
du -sh /path/to/VLA-master-thesis-2026/emil
```

---

## Troubleshooting

### Docker Build Issues

```bash
# Clear Docker cache and rebuild
docker system prune -a
docker build --no-cache -t flower_vla_calvin:latest .
```

### GPU Out of Memory

```bash
# Reduce batch size in config
# Edit conf/config_calvin.yaml:
# batch_size: 32  # Instead of 64

# Or use gradient accumulation
```

### Training Crashes

```bash
# Check logs
tail -n 100 logs/training_*/training.log

# Check GPU status
nvidia-smi

# Verify disk space
df -h
```

### SSH Connection Drops

```bash
# Use tmux/screen to keep session alive
tmux new -s flower_training

# Configure SSH to keep alive
# Add to ~/.ssh/config:
# Host cluster.address.com
#     ServerAliveInterval 60
#     ServerAliveCountMax 3
```

---

## Expected Training Time

With 2x NVIDIA A6000 (48GB each):
- **Epoch time:** ~30-60 minutes per epoch (depends on dataset size)
- **Full training (100 epochs):** ~50-100 hours
- **Debug dataset (1 epoch):** ~5-10 minutes

**Recommendation:** Start with debug dataset and 1-2 epochs to verify everything works, then run full training.

---

## Post-Training Checklist

- [ ] Model checkpoints saved and backed up
- [ ] Training logs exported from WandB
- [ ] Configuration files saved
- [ ] Results documented
- [ ] Cluster storage cleaned up
- [ ] Dataset removed (if not needed)
- [ ] Docker images cleaned up (if not needed)
- [ ] VPN disconnected

---

## Quick Reference Commands

```bash
# Connect to cluster
ssh your_username@cluster.address.com

# Check GPUs
nvidia-smi

# Start training
tmux new -s flower
./docker_run.sh
./run_training_cluster.sh

# Detach from tmux
Ctrl+B, then D

# Reattach to training
tmux attach -t flower

# Monitor logs
tail -f logs/training_*/training.log

# Stop training
Ctrl+C (inside Docker)

# Clean up
docker system prune -a
rm -rf dataset/calvin_debug_dataset
```

---

## Additional Resources

- WandB Dashboard: https://wandb.ai/VLA-Thesis/calvin_a6000
- CALVIN Dataset: https://github.com/mees/calvin
- Florence-2 Model: https://huggingface.co/microsoft/Florence-2-large

---

**Created:** 2026-02-02
**Last Updated:** 2026-02-02
