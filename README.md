# FLOWER VLA — MeanFlow / iMF Decoder Variants

This repository extends [FLOWER VLA](https://www.arxiv.org/pdf/2509.04996)
(Reuss et al., CoRL 2025) with single-step **MeanFlow** and **iMF**
(improved MeanFlow with proprioception) decoder variants, and adds full
LIBERO training/evaluation, Leonardo HPC scripts, and a Docker setup.

This is the codebase for the master's thesis fork; the upstream FLOWER paper,
weights, and pretraining code live at
[intuitive-robots/flower_vla_calvin](https://github.com/intuitive-robots/flower_vla_calvin)
and [intuitive-robots/flower_vla_pret](https://github.com/intuitive-robots/flower_vla_pret).

---

## Model Variants

All variants share the Florence-2-large VLM backbone and the same DiT action
decoder structure; they differ in the loss / sampling regime.

| `model=` | Class | Sampling steps | Loss | Notes |
|---|---|---|---|---|
| `flower` | `flower.models.flower.FLOWERVLA` | 4 | Rectified flow | Upstream FLOWER baseline. |
| `meanflower` | `flower.models.meanflower.MeanFlowerVLA` | 1 | Mean Flow (logit-normal noise) | Single-step variant; meant to match FLOWER quality at 4× faster inference. |
| `imf` | `flower.models.meanflower.MeanFlowerVLA` | 1 | Mean Flow + proprioception | `meanflower` with `use_proprioception=True`; the decoder receives joint state alongside image features. Best LIBERO numbers in this fork. |
| `decoupled_meanflower` | `flower.models.meanflower.MeanFlowerVLA` | 1 | Decoupled Mean Flow | Splits DiT blocks into a `t`-conditioned encoder and an `r`-conditioned decoder; experimental ([arxiv 2510.24474](https://arxiv.org/abs/2510.24474)). |

Configs live in `conf/model/{flower,meanflower,imf,decoupled_meanflower}.yaml`.
Architecture details for the MeanFlow variants are in
[`docs/MEANFLOW.md`](docs/MEANFLOW.md) and
[`docs/flower_meanflow_architecture.md`](docs/flower_meanflow_architecture.md).

---

## Results (LIBERO, best epoch)

| Benchmark | FLOWER | iMF |
|---|---|---|
| LIBERO-Spatial | 0.988 | 0.979 |
| LIBERO-Object | 1.000 | 0.984 |
| LIBERO-Goal | 0.975 | 1.000 |
| LIBERO-10 | 0.967 | 0.925 |

Per-epoch and per-task breakdowns: [`results.md`](results.md).

---

## Installation

The submodules (`calvin_env`, `LIBERO`) are required. Clone with
`--recurse-submodules`:

```bash
git clone --recurse-submodules <this-repo-url> flower_vla_calvin
cd flower_vla_calvin
```

Pick **one** of the following install paths.

### A. Conda (local, 4× GPU)

```bash
conda create -n flower_cal python=3.9
conda activate flower_cal

# tacto (calvin_env dependency)
cd calvin_env/tacto && pip install -e . && cd ..
pip install -e .                              # calvin_env
cd ..

# LIBERO
cd LIBERO
pip install -r requirements.txt
pip install -e .
pip install numpy~=1.23
cd ..

# pyhash (vendored — upstream wheels are flaky on slurm)
pip install setuptools==57.5.0
cd pyhash-0.9.3 && python setup.py build && python setup.py install && cd ..

# Project deps
pip install -r requirements.txt
pip install -e .
```

### B. Docker

```bash
./scripts/docker_build.sh                     # builds flower_vla_calvin:latest
./scripts/docker_run.sh                       # starts container with GPUs + repo mounted
```

The Dockerfile pins CUDA 12.9 + PyTorch 2.10 and sets `MUJOCO_GL=egl` for
headless rendering.

### C. Leonardo HPC (CINECA)

See [`docs/LEONARDO_HPC.md`](docs/LEONARDO_HPC.md). One-time setup:
`./scripts/leonardo/setup_leonardo.sh all` provisions the venv, downloads
CALVIN + LIBERO into `$LEONARDO_WORK/data/`, and pre-caches Florence-2 (compute
nodes are offline).

---

## Data

### CALVIN

```bash
cd dataset
sh download_data.sh ABCD          # or: D, debug
cd ..
```

CALVIN training reads action chunks across ~10 `episode_*.npz` files per
sample. Without preprocessing this saturates disk bandwidth (~2 GB/iter).
Run the extractor once per split:

```bash
python preprocess/extract_by_key.py \
    --in_root ./dataset \
    --in_task task_ABCD_D \
    --extract_key rel_actions
```

### LIBERO

```bash
cd LIBERO
python benchmark_scripts/download_libero_datasets.py --use-huggingface
cd ..
```

---

## Pretrained Checkpoint

The FLOWER pretrained backbone (1.67 GB, 360k steps) is on HuggingFace:
[`mbreuss/flower_vla_pret`](https://huggingface.co/mbreuss/flower_vla_pret).
Other variants are in the
[FLOWER collection](https://huggingface.co/collections/mbreuss/flower-vla-67d60e95bf2990699fcef81f).

Download `360000_model_weights.pt` to:

```
checkpoints/pretrained/360000_model_weights.pt
```

This is the default location read by `conf/model/flower.yaml` when
`load_pretrained=True`.

---

## Training

Both entrypoints use Hydra; any value in `conf/` can be overridden on the
command line.

### CALVIN

```bash
python flower/training_calvin.py \
    model=flower \
    datamodule=calvin \
    benchmark_name=calvin_abcd \
    root_data_dir=./dataset/task_ABCD_D \
    use_extracted_rel_actions=true \
    devices=4 batch_size=8 max_epochs=40
```

For replication of the published FLOWER numbers: 4× GPU, batch_size 8, ~40
epochs, evaluation around epoch 19 — see `conf/config_calvin.yaml`.

### LIBERO

Pick the model and benchmark:

```bash
# FLOWER baseline (4-step)
python flower/training_libero.py \
    model=flower \
    libero_benchmark=libero_spatial \
    root_data_dir=./LIBERO/libero/datasets/libero_spatial \
    devices=4 batch_size=8

# MeanFlow (1-step)
python flower/training_libero.py model=meanflower libero_benchmark=libero_object ...

# iMF (1-step + proprio) — pretrained checkpoint recommended
python flower/training_libero.py \
    model=imf \
    libero_benchmark=libero_10 \
    +pretrain_chk=./checkpoints/pretrained/imf_checkpoint_290000.safetensors \
    devices=4 batch_size=8
```

Evaluation rollouts run every `callbacks.rollout_lh.rollout_freq` × 1k steps,
starting after `rollout_lh_skip_epochs`.

---

## Evaluation

### CALVIN

```bash
./scripts/run_evaluation.sh
```

Expects:
- `checkpoints/calvin_d/model.safetensors`
- `dataset/calvin_debug_dataset/`

Output goes to `evaluation/calvin_debug_evaluation/`.

### LIBERO

```bash
./scripts/run_libero_evaluation.sh <benchmark> <model>
# e.g.
./scripts/run_libero_evaluation.sh libero_spatial imf
./scripts/run_libero_evaluation.sh libero_10 flower
```

`<benchmark>` ∈ {`libero_spatial`, `libero_object`, `libero_goal`,
`libero_10`, `libero_90`}.
`<model>` ∈ {`flower`, `meanflower`, `imf`, `decoupled_meanflower`}.

Expected checkpoint layout (the script auto-detects `.ckpt` or
`.safetensors`):

```
checkpoints/
└── <model>/
    └── <benchmark>/
        ├── .hydra/config.yaml
        └── model.ckpt
```

`num_sampling_steps` is set automatically (4 for `flower`, 1 for the rest).

---

## HPC (Leonardo)

Cluster-specific setup, submodules, sbatch templates, wandb-offline sync,
and rsync recipes for moving checkpoints on/off the scratch FS are in
[`docs/LEONARDO_HPC.md`](docs/LEONARDO_HPC.md). Production sbatch scripts:

- `scripts/leonardo/sbatch_train_calvin.sh`
- `scripts/leonardo/sbatch_train_libero.sh`
- `scripts/leonardo/sbatch_train_libero_imf.sh`

Debug variants and one-off conversion utilities are under
[`tools/`](tools/README.md).

---

## Repository Layout

```
flower_vla_calvin/
├── flower/                    # Source: models, datasets, training, eval, rollouts
│   ├── models/                #   flower.py, meanflower.py
│   ├── datasets/              #   CALVIN (HULC) + LIBERO data modules
│   ├── training_calvin.py     #   CALVIN training entrypoint
│   ├── training_libero.py     #   LIBERO training entrypoint
│   └── evaluation/            #   flower_evaluate.py (CALVIN), flower_eval_libero.py
├── conf/                      # Hydra configs
│   ├── config_{calvin,libero}.yaml
│   ├── eval_{calvin,libero}.yaml
│   ├── model/                 #   flower / meanflower / imf / decoupled_meanflower
│   └── datamodule/, callbacks/, annotations/
├── scripts/                   # Runnable scripts
│   ├── run_evaluation.sh, run_libero_evaluation.sh
│   ├── download_*.sh          #   data + pretrained checkpoints
│   └── leonardo/              #   HPC sbatch scripts
├── docs/                      # LEONARDO_HPC.md + MeanFlow architecture refs
├── preprocess/                # CALVIN action-chunk extractor
├── tools/                     # One-off utilities (ckpt conversion, debug jobs)
├── calvin_env/, LIBERO/       # Submodules
├── pyhash-0.9.3/              # Vendored pyhash (slurm install workaround)
├── Dockerfile, docker-compose.yml
└── results.md                 # Per-epoch / per-task LIBERO results
```

---

## Known Issues

- `calvin_env`'s `play_table_env.py` calls
  `get_git_commit_hash(Path(calvin_env.__file__))` at import time. In some
  installs `calvin_env.__file__` is `None`, which raises. The submodule
  pinned in this repo (`849b4c3`) includes a `None` guard for that path.
  If you point at a different `calvin_env`, you may need to delete that
  log line manually
  ([upstream reference](https://github.com/mees/calvin_env/blob/797142c588c21e76717268b7b430958dbd13bf48/calvin_env/envs/play_table_env.py#L72)).
- LIBERO's `numpy` constraint conflicts with PyTorch 2.10. Install
  `numpy~=1.23` *after* `pip install -e .` inside the LIBERO submodule
  (the order in [Installation §A](#a-conda-local-4-gpu) handles this).

---

## Acknowledgements

This work builds on:
- **FLOWER** — [intuitive-robots/flower_vla_calvin](https://github.com/intuitive-robots/flower_vla_calvin)
  ([CoRL 2025 paper](https://www.arxiv.org/pdf/2509.04996)).
- **CALVIN** — [mees/calvin](https://github.com/mees/calvin) (MIT).
- **LIBERO** — [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (MIT).
- **HULC** — [lukashermann/hulc](https://github.com/lukashermann/hulc) (MIT).
- **mimictest** — [EDiRobotics/mimictest](https://github.com/EDiRobotics/mimictest) (Apache-2.0).

## Citation

Upstream FLOWER:

```bibtex
@inproceedings{reuss2025flower,
  title  = {{FLOWER}: Democratizing Generalist Robot Policies with Efficient Vision-Language-Flow Models},
  author = {Moritz Reuss and Hongyi Zhou and Marcel R{\"u}hle and {\"O}mer Erdin{\c{c}} Ya{\u{g}}murlu and Fabian Otto and Rudolf Lioutikov},
  booktitle = {9th Annual Conference on Robot Learning},
  year   = {2025},
  url    = {https://openreview.net/forum?id=JeppaebLRD}
}
```
