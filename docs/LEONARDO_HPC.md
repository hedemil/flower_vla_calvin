# Leonardo HPC (CINECA) — Training Guide

Guide for running FLOWER VLA training on Leonardo using a Python venv and SLURM.

**Key constraints:** Compute nodes have **no internet**, use SLURM batch scheduling, and have 4x A100 GPUs per node.

---

# Re-generate certificate after 12 hrs
```bash
step ssh login 'emhed@kth.se' --provisioner cineca-hpc
```

## Login to LEONARDO
```bash
ssh ehed0000@login.leonardo.cineca.it
```

## Prerequisites

- Access to Leonardo with a valid compute account
- `LEONARDO_FAST` (or `$FAST`) and `LEONARDO_WORK` (or `$WORK`) environment variables set
- WandB API key (for syncing offline runs)

---

## 1. One-Time Setup (Login Node)

```bash
# Set your storage paths
export LEONARDO_FAST=/leonardo_scratch/fast/<YOUR_ACCOUNT>
export LEONARDO_WORK=/leonardo_work/<YOUR_ACCOUNT>

# .venv
source $LEONARDO_WORK/project/venvs/flowervla/bin/activate
cd $LEONARDO_FAST/project/flower_vla_calvin

python -c "import torch; print('PyTorch:', torch.__version__); import triton; print('Triton:', triton.__version__); import jvp_flash_attention; print('jvp_flash_attention: installed')" 

# Rsync the repo (including submodules) to fast storage
rsync -av --exclude='.git' --exclude='dataset' --exclude='logs' \
    ./flower_vla_calvin/ leonardo:$FAST/flower_vla_calvin/

# SSH to Leonardo and run full setup
ssh leonardo
cd $FAST/flower_vla_calvin
./scripts/leonardo/setup_leonardo.sh all
```

Or run individual steps:
```bash
./scripts/leonardo/setup_leonardo.sh dirs      # Create directory structure + sync code
./scripts/leonardo/setup_leonardo.sh venv      # Create venv + install all deps
./scripts/leonardo/setup_leonardo.sh calvin    # Download + preprocess CALVIN
./scripts/leonardo/setup_leonardo.sh libero    # Download LIBERO datasets
./scripts/leonardo/setup_leonardo.sh hf_cache  # Pre-cache Florence-2 model
```

### What each step does

| Step | Internet needed? | What it does |
|------|:---:|---|
| `dirs` | No | Creates dir structure on `$FAST` and `$WORK`, syncs code |
| `venv` | Yes | Creates Python venv, installs all pip deps + submodules |
| `calvin` | Yes (wget) | Downloads `task_ABC_D` + extracts `rel_actions` |
| `libero` | Yes (HuggingFace) | Downloads all LIBERO benchmarks via HuggingFace |
| `hf_cache` | Yes | Pre-caches Florence-2-large for offline compute |

---

## 2. Directory Structure

```
$FAST/flower_vla_calvin/              # Code (rsynced/cloned, fast I/O)
├── conf/                             # Hydra configs
├── flower/                           # Source code
├── logs/                             # Training logs + Hydra outputs
├── checkpoints/                      # Model checkpoints
└── wandb_runs/                       # WandB offline runs

$WORK/venvs/flower_vla_calvin/        # Python virtual environment
$WORK/data/
├── calvin/task_ABC_D/                 # CALVIN dataset
└── libero/libero_spatial/            # LIBERO datasets
$WORK/hf_cache/                       # Pre-cached Florence-2 model
```

**Storage tiers:**
- `$FAST` — fast scratch, for code/logs/checkpoints (high IOPS, not persistent long-term)
- `$WORK` — persistent, for venv/data/model cache (slower, quota-managed)

---

## 3. Fill In Your Account

Before submitting jobs, set your account in the SLURM scripts:

```bash
sed -i 's/<YOUR_ACCOUNT>/your_actual_account/g' scripts/leonardo/sbatch_*.sh
```

---

## 4. Validate the Setup (Run First!)

Before launching a long run, do a short smoke test on the debug QoS: submit a
production training script with a tight time/epoch cap and confirm it reaches
the first training steps.

```bash
sbatch --qos=boost_qos_dbg --time=00:30:00 \
    scripts/leonardo/sbatch_train_libero.sh meanflower max_epochs=1
```

Then check the output (`cat <job-name>_<JOBID>.out`) to confirm: GPUs visible,
Florence-2 loads from the offline HF cache, data dirs are accessible, and
training reaches the first steps end-to-end.

---

## 5. Production Training

### LIBERO

```bash
# Default (libero_spatial, 4 GPUs, meanflower)
sbatch scripts/leonardo/sbatch_train_libero.sh

# With overrides
sbatch scripts/leonardo/sbatch_train_libero.sh libero_benchmark=libero_goal batch_size=8

# Different model
sbatch scripts/leonardo/sbatch_train_libero.sh model=imf

# From pretrained checkpoint
sbatch scripts/leonardo/sbatch_train_libero.sh \
    model=flower \
    model.load_pretrained=True \
    model.pretrained_model_path=$FAST/flower_vla_calvin/checkpoints/pretrained/360000_model_weights.pt
```

### CALVIN

```bash
# Default (task_ABC_D, calvin_abcd, 4 GPUs, meanflower)
sbatch scripts/leonardo/sbatch_train_calvin.sh

# With overrides
sbatch scripts/leonardo/sbatch_train_calvin.sh batch_size=8 max_epochs=30

# D-only dataset (download task_D_D first)
sbatch scripts/leonardo/sbatch_train_calvin.sh \
    root_data_dir=$WORK/data/calvin/task_D_D \
    benchmark_name=calvin_d
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

## 6. Monitor Jobs

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

## 7. Sync WandB Runs

Training runs in `WANDB_MODE=offline` (no internet on compute). Sync from the login node:

```bash
# Set your API key
export WANDB_API_KEY="your_key_here"

# Sync all offline runs
./scripts/leonardo/sync_wandb.sh
```

Runs appear at: `https://wandb.ai/VLA-Thesis/<project>`

---

## 8. Iterating Code

Edit code directly on `$FAST` and re-submit — no rebuilds needed:

```bash
# Edit on Leonardo
vim $FAST/flower_vla_calvin/conf/config_libero.yaml
vim $FAST/flower_vla_calvin/flower/training_libero.py

# Re-submit
sbatch scripts/leonardo/sbatch_train_libero.sh
```

To sync local changes from your machine:
```bash
rsync -av --delete flower/  leonardo:$FAST/flower_vla_calvin/flower/
rsync -av --delete conf/    leonardo:$FAST/flower_vla_calvin/conf/
```

---

## 9. Download checkpoint
```bash
rsync -avP leonardo:/leonardo_scratch/fast/AIFAC_F02_024/flower_vla_calvin/logs/runs/<path to checkpoint> ./<download dir>

rsync -avP leonardo:/leonardo_scratch/fast/AIFAC_F02_024/flower_vla_calvin/logs/runs/<path to run>/.hydra ./<path to checkpoint>

# Download epoch 109 checkpoint Flower libero_10 %
rsync -avP leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-14/flower_39829244/saved_models/epoch=109_eval_lh checkpoints/flower/libero_10/

# Download hydra config Flower libero_10 %
rsync -avP leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-14/flower_39829244/.hydra checkpoints/flower/libero_10/

# Download best checkpoint (epoch 99) Flower libero_spatial %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-13/flower_39763611/saved_models/epoch=99_eval_lh" checkpoints/flower/libero_spatial/

# Download hydra config Flower libero_spatial %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-13/flower_39763611/.hydra" checkpoints/flower/libero_spatial/

# Download best checkpoint (epoch 89) iMF libero_spatial %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-14/imf_39829339/saved_models/epoch=89_eval_lh" checkpoints/meanflower/libero_spatial/

# Download hydra config iMF libero_spatial %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-14/imf_39829339/.hydra" checkpoints/meanflower/libero_spatial/

# Download best checkpoint (epoch 109) Flower libero_goal %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-15/flower_39877274/saved_models/epoch=109_eval_lh" checkpoints/flower/libero_goal/

# Download hydra config Flower libero_goal %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-15/flower_39877274/.hydra" checkpoints/flower/libero_goal/

# Download best checkpoint (epoch 29) Flower libero_object %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-15/flower_39921960/saved_models/epoch=29_eval_lh" checkpoints/flower/libero_object/

# Download hydra config Flower libero_object %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-15/flower_39921960/.hydra" checkpoints/flower/libero_object/

# Download best checkpoint (epoch 89) iMF libero_10 %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-15/imf_39878497/saved_models/epoch=89_eval_lh" checkpoints/meanflower/libero_10/

# Download hydra config iMF libero_10 %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-15/imf_39878497/.hydra" checkpoints/meanflower/libero_10/

# Download best checkpoint (epoch 99) iMF libero_goal %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-15/imf_39908191/saved_models/epoch=19_eval_lh" checkpoints/meanflower/libero_goal/

# Download hydra config iMF libero_goal %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-15/imf_39908191/.hydra" checkpoints/meanflower/libero_goal/

# Download best checkpoint (epoch 109) iMF libero_spatial %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-15/imf_39921929/saved_models/epoch=109_eval_lh" checkpoints/meanflower/libero_spatial/

# Download hydra config iMF libero_spatial %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-15/imf_39921929/.hydra" checkpoints/meanflower/libero_spatial/

# Download best checkpoint (epoch 119) iMF libero_object %
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-17/imf_40050147/saved_models/epoch=119_eval_lh" checkpoints/meanflower/libero_object/

# Download hydra config iMF libero_object
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-04-17/imf_40050147/.hydra" checkpoints/meanflower/libero_object/
```
## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Venv not found` | Run `setup_leonardo.sh venv` on login node first |
| `Florence-2 load fails` | Run `setup_leonardo.sh hf_cache` on login node first |
| `LIBERO import error` | Check `~/.libero/config.yaml` exists with correct paths |
| Port conflict (`MASTER_PORT`) | Script auto-randomizes; if issues, set manually |
| OOM on A100 | Reduce `batch_size=2` or `accumulate_grad_batches=8` |
| `NCCL timeout` | Check `NCCL_NET=IB` and `NCCL_IB_HCA=mlx5` are set |
| `No space left on device` | Clean old logs: `rm -rf $FAST/flower_vla_calvin/logs/runs/old_date/` |
| Job pending in queue | Use `--qos=boost_qos_dbg` for quick tests (30 min max) |
| MuJoCo EGL rendering fails | Check `MUJOCO_GL=egl` is set; may need `module load mesa` |
| pyhash build fails | Ensure `build-essential` / `gcc` available; use `setuptools==57.5.0` |

---

## Scripts Reference

| Script | Description |
|--------|-------------|
| `scripts/leonardo/setup_leonardo.sh` | One-time setup (dirs, venv, data, model cache) |
| `scripts/leonardo/download_hf_models.py` | Pre-cache Florence-2 for offline use |
| `scripts/leonardo/sbatch_train_libero.sh` | Production LIBERO training (24h) |
| `scripts/leonardo/sbatch_train_calvin.sh` | Production CALVIN training (24h) |
| `scripts/leonardo/sbatch_train_libero_imf.sh` | Production LIBERO iMF training |
| `scripts/leonardo/sync_wandb.sh` | Sync offline WandB runs |
