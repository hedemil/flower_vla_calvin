# Leonardo HPC (CINECA) — Training Guide

Guide for running FLOWER VLA training on Leonardo using Singularity containers and SLURM.

**Key constraints:** Compute nodes have **no internet**, use SLURM batch scheduling, and have 4x A100 GPUs per node.

---

## Prerequisites

- Access to Leonardo with a valid compute account
- `LEONARDO_FAST` (or `$FAST`) and `LEONARDO_WORK` (or `$WORK`) environment variables set
- Docker image `flower_vla_calvin:latest` built locally (see `docs/SETUP_GUIDE.md`)
- WandB API key (for syncing offline runs)

---

## 1. Build & Transfer Singularity Container

The project uses a Singularity container converted from the existing Docker image. This gives the exact same environment (MuJoCo 2.3.7, pyhash, calvin_env, LIBERO, PyTorch 2.2.2 + CUDA 11.8).

**Option A — Via container registry:**
```bash
# Local machine: push Docker image to registry
docker tag flower_vla_calvin:latest ghcr.io/<YOUR_USER>/flower_vla_calvin:latest
docker push ghcr.io/<YOUR_USER>/flower_vla_calvin:latest

# Leonardo login node: pull as Singularity
singularity pull --dir $WORK/containers/ docker://ghcr.io/<YOUR_USER>/flower_vla_calvin:latest
mv $WORK/containers/flower_vla_calvin_latest.sif $WORK/containers/flower_vla_calvin.sif
```

**Option B — Direct build + transfer:**
```bash
# Local machine (needs Singularity installed):
singularity build flower_vla_calvin.sif docker-daemon://flower_vla_calvin:latest

# Transfer to Leonardo (~10GB):
rsync -avP flower_vla_calvin.sif leonardo:$WORK/containers/
```

---

## 2. One-Time Setup (Login Node)

```bash
# Sync the repo to Leonardo
rsync -av --exclude='.git' --exclude='dataset' --exclude='logs' \
    ./flower_vla_calvin/ leonardo:$FAST/flower_vla_calvin/

# SSH to Leonardo
ssh leonardo

# Run full setup (creates dirs, downloads data, caches models)
cd $FAST/flower_vla_calvin
./scripts/leonardo/setup_leonardo.sh all
```

Or run individual steps:
```bash
./scripts/leonardo/setup_leonardo.sh dirs      # Create directory structure
./scripts/leonardo/setup_leonardo.sh calvin    # Download + preprocess CALVIN
./scripts/leonardo/setup_leonardo.sh libero    # Download LIBERO datasets
./scripts/leonardo/setup_leonardo.sh hf_cache  # Pre-cache Florence-2 model
```

### What each step does

| Step | Internet needed? | Container needed? | What it does |
|------|:---:|:---:|---|
| `dirs` | No | No | Creates dir structure on `$FAST` and `$WORK`, syncs code |
| `calvin` | Yes (wget) | Yes (preprocessing) | Downloads `task_D_D` + extracts `rel_actions` |
| `libero` | Yes (HuggingFace) | Yes | Downloads `libero_spatial` via HuggingFace |
| `hf_cache` | Yes | Yes | Pre-caches Florence-2-large for offline compute |

---

## 3. Directory Structure

```
$FAST/flower_vla_calvin/              # Code (rsynced, fast I/O)
├── conf/                             # Hydra configs (bind-mounted for iteration)
├── flower/                           # Source code (bind-mounted for iteration)
├── logs/                             # Training logs + Hydra outputs
├── checkpoints/                      # Model checkpoints
├── wandb_runs/                       # WandB offline runs
└── scripts/leonardo/                 # SLURM scripts

$WORK/containers/flower_vla_calvin.sif  # Singularity image (~10GB)
$WORK/data/
├── calvin/task_D_D/                    # CALVIN dataset
└── libero/libero_spatial/              # LIBERO datasets
$WORK/hf_cache/                         # Pre-cached Florence-2 model
```

**Storage tiers:**
- `$FAST` — fast scratch, for code/logs/checkpoints (high IOPS, not persistent long-term)
- `$WORK` — persistent, for large data/container (slower, quota-managed)

---

## 4. Fill In Your Account

Before submitting jobs, set your account in the SLURM scripts:

```bash
# Replace <YOUR_ACCOUNT> in all sbatch scripts
sed -i 's/<YOUR_ACCOUNT>/your_actual_account/g' scripts/leonardo/sbatch_*.sh
```

---

## 5. Debug Job (Run First!)

Always validate the setup before launching a long training:

```bash
sbatch scripts/leonardo/sbatch_debug.sh libero
# or
sbatch scripts/leonardo/sbatch_debug.sh calvin
```

This 30-minute job:
1. Prints GPU info (`nvidia-smi`)
2. Checks Python, PyTorch, CUDA versions inside container
3. Verifies Florence-2 loads from cache (offline)
4. Checks data directories are accessible
5. Runs 10 training batches to confirm end-to-end

Check the output:
```bash
cat flower-debug_<JOBID>.out
```

---

## 6. Production Training

### LIBERO

```bash
# Default (libero_spatial, 4 GPUs, meanflower)
sbatch scripts/leonardo/sbatch_train_libero.sh

# With overrides
sbatch scripts/leonardo/sbatch_train_libero.sh libero_benchmark=libero_goal batch_size=8

# Different model
sbatch scripts/leonardo/sbatch_train_libero.sh model=decoupled_meanflower

# From pretrained checkpoint
sbatch scripts/leonardo/sbatch_train_libero.sh \
    model=decoupled_meanflower \
    model.load_pretrained=True \
    model.pretrained_model_path=/workspace/flower_vla_calvin/checkpoints/pretrained/360000_model_weights.pt
```

### CALVIN

```bash
# Default (task_D_D, calvin_d, 4 GPUs, meanflower)
sbatch scripts/leonardo/sbatch_train_calvin.sh

# With overrides
sbatch scripts/leonardo/sbatch_train_calvin.sh batch_size=8 max_epochs=30

# ABCD dataset (download task_ABCD_D first)
sbatch scripts/leonardo/sbatch_train_calvin.sh \
    root_data_dir=/workspace/flower_vla_calvin/dataset/task_ABCD_D \
    benchmark_name=calvin_abcd
```

All scripts accept arbitrary Hydra overrides via `$@`.

### SLURM Configuration

Both production scripts use:
```
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # PL spawns DDP internally
#SBATCH --gpus-per-node=4       # 4x A100
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
```

**Why `ntasks-per-node=1`?** PyTorch Lightning auto-detects SLURM environment variables and spawns 4 DDP processes itself. Using `ntasks=4` + `srun` would conflict and double-spawn.

---

## 7. Monitor Jobs

```bash
# Check queue
squeue -u $USER

# Watch output in real-time
tail -f flower-libero_<JOBID>.out

# Cancel a job
scancel <JOBID>

# Check job efficiency after completion
seff <JOBID>
```

---

## 8. Sync WandB Runs

Training runs in `WANDB_MODE=offline` (no internet on compute). Sync from the login node:

```bash
# Set your API key
export WANDB_API_KEY="your_key_here"

# Sync all offline runs
./scripts/leonardo/sync_wandb.sh
```

Runs appear at: `https://wandb.ai/VLA-Thesis/<project>`

---

## 9. Iterating Code Without Rebuilding

The `conf/` and `flower/` directories are bind-mounted from `$FAST`, overlaying the container's baked copies. Edit code on the fast filesystem:

```bash
# Edit config
vim $FAST/flower_vla_calvin/conf/config_libero.yaml

# Edit training code
vim $FAST/flower_vla_calvin/flower/training_libero.py

# Re-submit — no container rebuild needed
sbatch scripts/leonardo/sbatch_train_libero.sh
```

To sync local changes from your machine:
```bash
rsync -av --delete flower/  leonardo:$FAST/flower_vla_calvin/flower/
rsync -av --delete conf/    leonardo:$FAST/flower_vla_calvin/conf/
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Container not found` | Transfer `.sif` to `$WORK/containers/` (see step 1) |
| `Florence-2 load fails` | Run `setup_leonardo.sh hf_cache` on login node first |
| `LIBERO import error` | Check `--no-home --env HOME=/appuser` is set (preserves baked config) |
| Port conflict (`MASTER_PORT`) | Script auto-randomizes; if issues, set manually |
| OOM on A100 | Reduce `batch_size=2` or `accumulate_grad_batches=8` |
| `NCCL timeout` | Check `NCCL_NET=IB` and `NCCL_IB_HCA=mlx5` are set |
| `No space left on device` | Clean old logs: `rm -rf $FAST/flower_vla_calvin/logs/runs/old_date/` |
| Job pending in queue | Use `--qos=boost_qos_dbg` for quick tests (30 min max) |
| Data not found in container | Check bind mounts match expected container paths |
| CUDA version mismatch | Container CUDA 11.8 + host CUDA 12.1 is fine (forward-compatible via `--nv`) |

---

## Scripts Reference

| Script | Description |
|--------|-------------|
| `scripts/leonardo/setup_leonardo.sh` | One-time setup (dirs, data, model cache) |
| `scripts/leonardo/download_hf_models.py` | Pre-cache Florence-2 for offline use |
| `scripts/leonardo/sbatch_train_libero.sh` | Production LIBERO training (24h) |
| `scripts/leonardo/sbatch_train_calvin.sh` | Production CALVIN training (24h) |
| `scripts/leonardo/sbatch_debug.sh` | Debug validation job (30 min) |
| `scripts/leonardo/sync_wandb.sh` | Sync offline WandB runs |
