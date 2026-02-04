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
# For quick testing:
./download_data.sh debug

# For Calvin D->D training (FULL dataset, ~50GB):
./download_data.sh D

# For Calvin ABC->D (~150GB):
./download_data.sh ABC

# For Calvin ABCD->D (full, ~200GB):
./download_data.sh ABCD

# Preprocess dataset (REQUIRED - do this for whichever dataset you downloaded)
./docker_run.sh

# For debug dataset:
python preprocess/extract_by_key.py -i /workspace/flower_vla_calvin/dataset --in_task calvin_debug_dataset --in_split all -k rel_actions

# For Calvin D:
python preprocess/extract_by_key.py -i /workspace/flower_vla_calvin/dataset --in_task task_D_D --in_split all -k rel_actions

# For Calvin ABC:
python preprocess/extract_by_key.py -i /workspace/flower_vla_calvin/dataset --in_task task_ABC_D --in_split all -k rel_actions

# For Calvin ABCD:
python preprocess/extract_by_key.py -i /workspace/flower_vla_calvin/dataset --in_task task_ABCD_D --in_split all -k rel_actions

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

# With DEBUG dataset (for testing):
./run_training_cluster.sh dataset=debug

# With CALVIN D dataset (for D->D training):
./run_training_cluster.sh dataset=D

# With CALVIN ABC dataset:
./run_training_cluster.sh dataset=ABC

# With CALVIN ABCD dataset:
./run_training_cluster.sh dataset=ABCD

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

**Checkpoints are automatically saved during training:**
- Location: `logs/training_TIMESTAMP/checkpoints/`
- `last.ckpt` - Most recent checkpoint
- `epoch=X-step=Y.ckpt` - Top 3 best checkpoints (by validation loss)

```bash
# Inside Docker: Stop with Ctrl+C (training saves checkpoint automatically)

# Find your checkpoint directory
ls -lh logs/training_*/checkpoints/

# Copy checkpoints to permanent storage (IMPORTANT - do this before cleanup!)
TIMESTAMP=20260204_123456  # Replace with your actual timestamp
mkdir -p ~/backups/calvin_d2d_${TIMESTAMP}
cp -r logs/training_${TIMESTAMP}/checkpoints ~/backups/calvin_d2d_${TIMESTAMP}/
cp conf/config_calvin.yaml ~/backups/calvin_d2d_${TIMESTAMP}/

# Verify backup
ls -lh ~/backups/calvin_d2d_${TIMESTAMP}/checkpoints/

# Download to local machine (on your local machine, NOT on cluster):
scp -r username@cluster:~/backups/calvin_d2d_TIMESTAMP ./local_checkpoints/

# Clean up cluster (only after confirming backups!)
docker system prune -a
rm -rf dataset/calvin_debug_dataset
rm -rf logs/training_*
```

**Checkpoint file sizes:**
- Each checkpoint: ~1.5-2GB (full model weights + optimizer state)
- Last + top 3 best: ~6-8GB total
- Make sure you have enough space before training!

---

## Training Settings for Calvin D→D

**Default configuration (in run_training_cluster.sh):**
- **Batch size:** 4 per GPU × 2 GPUs × 4 grad accumulation = **32 effective batch size**
- **GPUs:** 2x A6000 (48GB each)
- **Epochs:** 25 (targets ~40k steps as per paper)
- **Strategy:** FSDP (memory-optimized multi-GPU)
- **Model:** Unfrozen Florence-2-large (full training)
- **Camera views:** Single (static only, to reduce memory)
- **Rollout evaluation:** Starts after epoch 20
- **EMA:** Starts at step 5000
- **Checkpointing:** Saves last + top 3 best by validation loss

**Why these settings?**
- Paper recommendation: 4 GPUs × batch_size=8 = 32 effective, trained for 40k steps, optimal at ~19 epochs
- Our setup matches the effective batch size (32) using gradient accumulation
- Single camera view reduces memory usage while maintaining performance
- 25 epochs should reach ~40k steps for Calvin D dataset

**Quick test run (5-10 min):**
```bash
# Debug run with limited batches
python flower/training_calvin.py \
    batch_size=2 \
    max_epochs=2 \
    trainer.limit_train_batches=10 \
    rollout_lh_skip_epochs=100 \
    callbacks.ema.start_step=100000
```

**Full training run:**
```bash
# D→D training (recommended for single environment)
./run_training_cluster.sh dataset=D

# ABC→D training (3 environments, longer training time)
./run_training_cluster.sh dataset=ABC max_epochs=30

# ABCD→D training (full dataset, follows paper exactly)
./run_training_cluster.sh dataset=ABCD max_epochs=30

# Override multiple parameters
./run_training_cluster.sh dataset=D max_epochs=30 rollout_skip=25

# Smaller batch if OOM occurs
./run_training_cluster.sh dataset=D batch_size=2
```

**Dataset comparison:**
| Dataset | Size | Environments | Recommended Epochs | Training Time (2×A6000) |
|---------|------|--------------|-------------------|------------------------|
| debug | ~1GB | D (small subset) | 2-5 (testing) | 1-2 hours |
| D | ~50GB | D only | 25 | 20-40 hours |
| ABC | ~150GB | A, B, C → D | 30 | 60-90 hours |
| ABCD | ~200GB | A, B, C, D → D | 30 | 80-120 hours |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM error | Reduce `batch_size=2` (default is 4, effective 32 with grad accum) |
| OOM during forward pass | Reduce `batch_size` or use single camera view (already default) |
| Build fails | `docker system prune -a` then rebuild |
| SSH drops | Use `tmux` or `screen` (CRITICAL for long training runs) |
| Slow download | Check VPN connection |
| Training crashes | Check `tail -f logs/training_*/training.log` |
| Validation error at start | Normal! Occurs once before training starts |
| No checkpoints saved | Check `logs/training_*/checkpoints/` - they save automatically |
| Rollout evaluation too frequent | Increase `rollout_skip=25` to evaluate less often |
| Disk space full | Checkpoints are ~2GB each. Clean old logs/checkpoints before training |

---

## Checkpoint Management Best Practices

**During training:**
- Checkpoints auto-save to `logs/training_TIMESTAMP/checkpoints/`
- `last.ckpt` updates every N steps (config-dependent)
- Top 3 best checkpoints kept based on validation loss
- Monitor disk space: `df -h`

**After training:**
1. **Immediately backup checkpoints** (before any cleanup)
2. Copy to permanent storage outside workspace
3. Verify backup integrity before deleting originals
4. Download to local machine for safety

**Checkpoint selection for evaluation:**
- Use `last.ckpt` for latest model state
- Use best checkpoint (lowest val/loss) for optimal performance
- Compare multiple top checkpoints if performance varies

**Example: Resume training from checkpoint**
```bash
python flower/training_calvin.py \
    resume_from_checkpoint=logs/training_TIMESTAMP/checkpoints/last.ckpt \
    max_epochs=35  # Continue training for more epochs
```

---

## Expected Times (2x A6000)

**Setup:**
- Docker build: 15-30 min (one-time)
- Dataset download:
  - debug: ~5 min
  - D: ~20-30 min
  - ABC: ~60-90 min
  - ABCD: ~90-120 min
- Dataset preprocessing: 5-15 min per dataset

**Training:**
| Dataset | Epochs | Steps/Epoch (approx) | Time per Epoch | Total Training Time |
|---------|--------|---------------------|----------------|---------------------|
| debug | 2 | 50-100 | 5-10 min | 10-20 min |
| D | 25 | 1600-2000 | 45-90 min | 20-40 hours |
| ABC | 30 | 4800-6000 | 2-3 hours | 60-90 hours |
| ABCD | 30 | 6400-8000 | 3-4 hours | 90-120 hours |

**Other:**
- Rollout evaluation: ~15-30 min per rollout
- Target total steps: ~40k (as per paper)
- Checkpoints saved automatically every N steps (configured in config)

---

**Pro Tips:**

1. **Always test first with limited batches:**
```bash
# Quick sanity check (5-10 min)
python flower/training_calvin.py \
    batch_size=2 max_epochs=1 \
    trainer.limit_train_batches=10 \
    rollout_lh_skip_epochs=100 \
    callbacks.ema.start_step=100000 \
    logger.name=sanity_check
```

2. **Monitor training in real-time:**
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Follow training logs
tail -f logs/training_*/training.log

# Check WandB for metrics
# https://wandb.ai/VLA-Thesis/calvin_a6000
```

3. **Backup checkpoints during training:**
   Even during training, checkpoints are being saved. You can copy them mid-training if needed.
