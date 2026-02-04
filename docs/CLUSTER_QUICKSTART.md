# Cluster Training Quick Start

Ultra-condensed guide for running on the cluster. See `CLUSTER_SETUP.md` for detailed instructions.

---

## 1. Connect & Setup (One-time)

```bash
# Connect to VPN


# SSH to cluster


# Clone repo WITH submodules (important!)
git clone --recurse-submodules https://github.com/hedemil/flower_vla_calvin.git
cd flower_vla_calvin
git checkout scripts/train

# Build Docker (15-30 min)
docker build -t flower_vla_calvin:latest .

# Download dataset
./download_data.sh debug

# Preprocess dataset (REQUIRED)
./docker_run.sh
python preprocess/extract_by_key.py -i /workspace/flower_vla_calvin/dataset --in_task calvin_debug_dataset --in_split all -k rel_actions
exit

# Set WandB key
export WANDB_API_KEY="your_key_here"
```

---

## 2. Start Training

```bash
# Start tmux (keeps training alive if SSH drops)
tmux new -s flower

# Launch Docker
./docker_run.sh

# Inside Docker: Run training
./run_training_cluster.sh

# Detach from tmux: Ctrl+B, then D
```

---

## 3. Monitor

```bash
# Reattach to tmux
tmux attach -t flower

# Or monitor GPU in new SSH session
watch -n 1 nvidia-smi

# View logs
tail -f logs/training_*/training.log

# Check WandB: https://wandb.ai/VLA-Thesis/calvin_a6000
```

---

## 4. After Training

```bash
# Inside Docker: Stop with Ctrl+C

# Copy checkpoints
mkdir -p ~/backups/flower_$(date +%Y%m%d)
cp saved_models/*.ckpt ~/backups/flower_$(date +%Y%m%d)/
cp conf/config_calvin.yaml ~/backups/flower_$(date +%Y%m%d)/

# Download to local machine
# (on local machine):
scp -r username@cluster:~/backups/flower_YYYYMMDD ./

# Clean up cluster
docker system prune -a
rm -rf dataset/calvin_debug_dataset
rm -rf logs/training_*
```

---

## Training Settings

**Default (in run_training_cluster.sh):**
- Batch size: 32 (16 per GPU with DDP)
- GPUs: 2x A6000
- Epochs: 100
- Unfrozen model (full training)
- Both camera views

**Why batch_size=32?** With unfrozen Florence-2-large (570M params) and dual camera views, the forward pass activation memory requires significant space. Batch_size=64 causes OOM during forward pass even on 47GB A6000s.

**To modify:** Edit `run_training_cluster.sh` or override parameters:

```bash
# Even smaller batch (if still OOM)
./run_training_cluster.sh batch_size=16

# Fewer epochs (for testing)
./run_training_cluster.sh max_epochs=10

# Single GPU
./run_training_cluster.sh devices=1
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM error | Reduce `batch_size=16` or `batch_size=8` (default is now 32) |
| Build fails | `docker system prune -a` then rebuild |
| SSH drops | Use `tmux` or `screen` |
| Slow download | Check VPN connection |
| Training crashes | Check `tail -f logs/training_*/training.log` |

---

## Expected Times (2x A6000)

- Docker build: 15-30 min
- Dataset download (debug): 5 min
- Training (1 epoch): 30-60 min
- Training (100 epochs): 50-100 hours

---

**Pro Tip:** Test with debug dataset and 1 epoch first to verify everything works before running full 100-epoch training.

```bash
# Quick test run (5-10 min)
./run_training_cluster.sh max_epochs=1 logger.name=test_run
```
