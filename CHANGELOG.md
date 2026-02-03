# Changelog
## 2026-02-03
- Added core Mean Flow implementation with time step conditioning
  - Added `MeanFlowDecoder` MLP with h (time step difference `t - r`) embedding
  - Added `_setup_dit_components_meanflow`, `decode_actions_meanflow`, `dit_forward_meanflow` functions
  - Added time step sampling functions for t and r according to Mean Flow constraints
  - Updated `docs/MEANFLOW.md` with detailed implementation notes (550 lines added)
  - Modified `flower/models/flower.py` and `flower/models/networks/transformers.py` (811 insertions)
- Merged training scripts PR (#4)
- Organized code structure

## 2026-02-02
- Added training scripts for local and cluster execution
  - Added `run_training.sh` for local training
  - Added `run_training_cluster.sh` with SLURM support
  - Added `CLUSTER_QUICKSTART.md` and `CLUSTER_SETUP.md` documentation
- Improved SLURM configuration for cluster training
- Updated `calvin_env` submodule
- Fixed download dataset script

## 2026-01-27
- Setup improvements with wandb logging support
  - Working evaluation for both LIBERO and Calvin with wandb logging
  - Updated Dockerfile and docker_run.sh for better volume mounting
  - Improved evaluation scripts with better wandb integration
- Merged setup PR (#3)

## 2026-01-26
- Fixed wandb logging bug in LIBERO evaluation - `wandb.log()` was being called even when `log_wandb=false`
- Added `log_wandb` parameter to `EvaluateLibero` class and wrapped wandb calls in conditional checks
- Successfully ran LIBERO spatial evaluation with 100% success rate (20/20 episodes)
- Measured inference performance: Latency 0.0120 s/step, Throughput 83.59 Hz

## 2026-01-23
- Added inference time measurement to Calvin evaluation (`flower/evaluation/flower_eval_libero.py`)

## 2026-01-22
- Added bfloat16 (bf16) autocast support for inference (~14-16% throughput improvement)
- Added `use_bf16` config option to `conf/eval_calvin.yaml`
- Modified `model.step()` in `flower/models/flower.py` to use `torch.autocast` when bf16 is enabled
- Added missing `conf` volume mount to `docker_run.sh` (config changes now reflected in container)

## 2026-01-21
- Added inference time measurement to Calvin evaluation (`flower/evaluation/flower_evaluate.py`)
- Fixed inference timing to only measure actual model calls, not cached actions (accounts for action chunking)
- Added effective per-action time reporting (inference_time / multistep)
- Added `@torch.no_grad()` to `model.step()` for faster inference (`flower/models/flower.py`)
- Added `torch.cuda.synchronize()` for accurate GPU timing measurement
- Added `torch.backends.cudnn.benchmark = True` for faster inference
- Fixed `calvin_env.__file__` being None in Docker (`calvin_env/calvin_env/envs/play_table_env.py`)
- Added `calvin_env` volume mount to `docker_run.sh` and `docker-compose.yml`

## 2026-01-19
- Added Dockerfile and docker-compose for running code
- Added scripts: `docker_build.sh`, `docker_run.sh`, `run_evaluation.sh`
- Added scripts: `download_calvin_checkpoint.sh`, `download_pretrained.sh`
- Added `install_nvidia_docker.sh` for NVIDIA Docker setup
- Minor fixes to Calvin evaluation (`flower/evaluation/flower_evaluate.py`, `utils.py`, `flower.py`, `rollout_video.py`)
