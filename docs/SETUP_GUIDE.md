# FLOWER VLA Setup Guide

This guide explains how to set up datasets and checkpoints for FLOWER VLA training and evaluation.

## Quick Start

### Option 1: Automated Setup (Recommended)

Run the setup script to automatically download and preprocess everything:

```bash
cd emil/
./setup_datasets.sh [CALVIN_SPLIT] [--skip-libero] [--skip-preprocessing]
```

**Arguments:**
- `CALVIN_SPLIT`: Choose dataset split (default: `ABCD`)
  - `D` - Single environment (task_D_D)
  - `ABC` - Three environments (task_ABC_D)
  - `ABCD` - Four environments (task_ABCD_D) - **Recommended for full training**
  - `debug` - Small debug dataset (already included)
- `--skip-libero`: Skip LIBERO dataset download (only download CALVIN)
- `--skip-preprocessing`: Skip CALVIN preprocessing (not recommended)

**Examples:**
```bash
# Full setup with ABCD split (recommended)
./setup_datasets.sh ABCD

# Only CALVIN with D split, skip LIBERO
./setup_datasets.sh D --skip-libero

# ABC split without preprocessing (for testing download only)
./setup_datasets.sh ABC false true
```

### Option 2: Manual Setup

If you prefer manual control, follow the steps below.

## Manual Setup Steps

### 1. Download Pretrained Checkpoint

```bash
cd emil/
./download_pretrained.sh
```

**Downloads:**
- HuggingFace repo: `mbreuss/flower_vla_pret`
- Files: `360000_model_weights.pt` (1.67 GB), `config.yaml`
- Location: `checkpoints/pretrained/`

### 2. Download CALVIN Dataset

```bash
cd dataset/
./download_data.sh [D|ABC|ABCD|debug]
cd ..
```

**Download source:** http://calvin.cs.uni-freiburg.de/dataset/

### 3. Preprocess CALVIN Dataset (CRITICAL!)

**Why preprocessing is needed:**
- FLOWER uses action chunking (~10 episode files per inference)
- Without preprocessing: ~2000MB/iteration disk bandwidth → reduced GPU utilization
- With preprocessing: Data consolidated into single `.npy` files → much faster I/O

**Run preprocessing:**
```bash
python preprocess/extract_by_key.py \
    --in_root $(pwd)/dataset \
    --in_task task_ABCD_D \
    --in_split all \
    --extract_key rel_actions
```

**Parameters:**
- `--in_root`: Path to dataset directory
- `--in_task`: Task folder (e.g., `task_D_D`, `task_ABC_D`, `task_ABCD_D`, or `all`)
- `--in_split`: Split to process (`training`, `validation`, or `all`)
- `--extract_key`: Data key to extract (default: `rel_actions`)
- `--force`: Overwrite existing extracted data

**Output:**
- File: `dataset/task_ABCD_D/training/extracted/ep_rel_actions.npy`
- Mapping: `dataset/task_ABCD_D/training/extracted/ep_npz_names.list`

### 4. Download LIBERO Datasets

```bash
cd LIBERO/

# Install LIBERO package
pip install -e .

# Download all datasets (recommended - uses HuggingFace mirror)
python benchmark_scripts/download_libero_datasets.py --use-huggingface

# Or download specific dataset
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_spatial \
    --use-huggingface

cd ..
```

**Available datasets:**
- `libero_spatial` (10 tasks) - Spatial reasoning
- `libero_object` (10 tasks) - Object interaction
- `libero_goal` (10 tasks) - Goal-oriented tasks
- `libero_90` (90 tasks) - Large benchmark
- `libero_10` (10 tasks) - Small benchmark

**Download sources:**
- `--use-huggingface`: HuggingFace Hub (`yifengzhu-hf/LIBERO-datasets`) - **Recommended**
- Default: Original Box URLs (may expire)

### 5. Download Calvin-Finetuned Checkpoint (Optional)

If you want to evaluate or continue training from a Calvin-finetuned model:

```bash
./download_calvin_checkpoint.sh [D|ABC|ABCD]
```

**Available checkpoints:**
- `mbreuss/flower_calvin_d` → `checkpoints/calvin_d/`
- `mbreuss/flower_calvin_abc` → `checkpoints/calvin_abc/`
- `mbreuss/flower_calvin_abcd` → `checkpoints/calvin_abcd/`

## Directory Structure After Setup

```
emil/
├── dataset/
│   ├── task_ABCD_D/                    # CALVIN dataset
│   │   ├── training/
│   │   │   ├── episode_0000000.npz
│   │   │   ├── ...
│   │   │   └── extracted/              # Preprocessed data
│   │   │       ├── ep_rel_actions.npy
│   │   │       └── ep_npz_names.list
│   │   ├── validation/
│   │   └── statistics.yaml
│   └── calvin_debug_dataset/           # Debug dataset (pre-included)
├── checkpoints/
│   ├── pretrained/                     # Pretrained FLOWER
│   │   ├── 360000_model_weights.pt
│   │   └── config.yaml
│   └── calvin_abcd/                    # Calvin-finetuned (optional)
│       └── ...
├── LIBERO/
│   ├── libero_spatial/
│   │   └── *.hdf5 (10 files)
│   ├── libero_object/
│   ├── libero_goal/
│   ├── libero_90/
│   └── libero_10/
├── outputs/                            # Training outputs
└── logs/                               # Training logs
```

## Docker Workflow

### 1. Build Docker Image

```bash
cd emil/
./docker_build.sh
```

Builds image: `flower_vla_calvin:latest`

### 2. Run Docker Container

```bash
./docker_run.sh
```

**What it does:**
- Mounts all necessary directories (dataset, checkpoints, configs, etc.)
- Enables GPU support (`--gpus all`)
- Sets up MuJoCo rendering (`MUJOCO_GL=egl`)
- Allocates shared memory (`--shm-size=16g`)
- Opens interactive bash shell

**Note:** The docker_run.sh script creates empty directories if they don't exist, so you can run it before downloading datasets. However, you'll need to download datasets from inside the container or exit and re-run after downloading.

### 3. Inside Container - Training

```bash
# CALVIN training
python flower/training.py \
    datamodule=calvin \
    datamodule.root_data_path=/workspace/flower_vla_calvin/dataset \
    model.calvin_dataset=task_ABCD_D

# LIBERO training
python flower/training.py \
    datamodule=libero \
    datamodule.datasets=[libero_spatial,libero_object]
```

### 4. Inside Container - Evaluation

```bash
# CALVIN evaluation
./run_evaluation.sh

# LIBERO evaluation
python flower/libero_evaluation.py \
    --model_path /workspace/flower_vla_calvin/checkpoints/pretrained/360000_model_weights.pt \
    --benchmark libero_spatial
```

## Verification Checklist

After setup, verify:

- [ ] Pretrained checkpoint exists: `checkpoints/pretrained/360000_model_weights.pt`
- [ ] CALVIN dataset downloaded: `dataset/task_ABCD_D/`
- [ ] CALVIN dataset preprocessed: `dataset/task_ABCD_D/training/extracted/ep_rel_actions.npy`
- [ ] LIBERO datasets downloaded: `LIBERO/libero_*/`
- [ ] Docker image built: `docker images | grep flower_vla_calvin`

## Troubleshooting

### Dataset Download Issues

**CALVIN download fails:**
- Check internet connection
- Verify http://calvin.cs.uni-freiburg.de/dataset/ is accessible
- Try a different split (debug is pre-included)

**LIBERO download fails (Box URLs):**
- Use `--use-huggingface` flag for alternative download source
- Box URLs may expire; HuggingFace mirror is more reliable

### Preprocessing Issues

**"No such file or directory" error:**
- Ensure CALVIN dataset is downloaded first
- Verify `--in_root` path points to `dataset/` directory
- Check `--in_task` matches your downloaded split

**Preprocessing is slow:**
- Normal for large datasets (ABCD can take 15-30 minutes)
- Progress is not shown, but it's working
- Check disk space (preprocessing creates additional files)

### Docker Issues

**GPU not available:**
- Install nvidia-docker2: `sudo apt-get install nvidia-docker2`
- Verify GPU: `nvidia-smi`
- Check Docker GPU support: `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

**Out of memory:**
- Increase `--shm-size` in docker_run.sh (currently 16g)
- Reduce batch size in training config

### Training Issues

**"FileNotFoundError: ep_rel_actions.npy":**
- Run preprocessing step (see Section 3 above)
- Or disable extracted data in config: set `use_extracted_rel_actions: false`

**Low GPU utilization:**
- Likely caused by disk I/O bottleneck
- **Solution:** Run preprocessing to consolidate episode data

## Additional Resources

- **Main README:** `README.md` - Project overview and performance metrics
- **Quickstart:** `QUICKSTART.md` - Quick training and evaluation guide
- **LIBERO README:** `LIBERO/README.md` - LIBERO-specific documentation
- **HuggingFace Collection:** https://huggingface.co/collections/mbreuss/flower-vla-67d60e95bf2990699fcef81f

## Disk Space Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Pretrained checkpoint | ~1.7 GB | Required |
| CALVIN (D split) | ~50 GB | Smallest dataset |
| CALVIN (ABC split) | ~150 GB | Medium dataset |
| CALVIN (ABCD split) | ~200 GB | Full dataset (recommended) |
| CALVIN preprocessed | +10-20% | Additional space for .npy files |
| LIBERO (all datasets) | ~50 GB | All 5 benchmarks |
| Docker image | ~15 GB | CUDA + PyTorch + dependencies |
| **Total (Full setup)** | **~300-350 GB** | ABCD + LIBERO + preprocessing |

**Recommendation:** Ensure at least 400GB free disk space for comfortable full setup.

## Dataset Information

### CALVIN Dataset Splits

| Split | Environments | Tasks | Training Episodes | Use Case |
|-------|-------------|-------|------------------|----------|
| D | 1 | 34 | ~100k | Fast training/testing |
| ABC | 3 | 34 | ~300k | Multi-environment |
| ABCD | 4 | 34 | ~400k | Full benchmark (recommended) |
| debug | 1 | 34 | ~100 | Quick testing (pre-included) |

### LIBERO Benchmarks

| Benchmark | Tasks | Episodes/Task | Total Episodes | Focus |
|-----------|-------|---------------|----------------|-------|
| libero_spatial | 10 | 50 | 500 | Spatial reasoning |
| libero_object | 10 | 50 | 500 | Object manipulation |
| libero_goal | 10 | 50 | 500 | Goal-oriented |
| libero_10 | 10 | 50 | 500 | General benchmark |
| libero_90 | 90 | 50 | 4500 | Large-scale benchmark |

## License & Citation

Please refer to the main `README.md` for citation information and license details.
