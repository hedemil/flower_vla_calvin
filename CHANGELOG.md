# Changelog
## 2026-01-27
- **CRITICAL FIX**: Updated MEANFLOW.md integration guide to use correct automatic differentiation method
  - Changed from `torch.autograd.grad` (backward-mode AD) to `torch.func.jvp` (forward-mode AD)
  - Added detailed implementation based on [official PyTorch Mean Flow implementation](https://github.com/Gsunshine/py-meanflow)
  - Added comprehensive notes on JVP usage, tangent vectors, and efficiency considerations
  - Added reference implementations and critical corrections section

## 2026-01-26
- Fixed wandb logging bug in LIBERO evaluation - `wandb.log()` was being called even when `log_wandb=false`
- Added `log_wandb` parameter to `EvaluateLibero` class and wrapped wandb calls in conditional checks
- Successfully ran LIBERO spatial evaluation
- Measured inference performance: Latency 0.0120 s/step, Throughput 83.59 Hz

## 2026-01-23
- Added inference time measurement to Calvin evaluation (`flower/evaluation/flower_evaluate.py`)
- Added inference time measurement to LIBERO evaluation (`flower/evaluation/flower_eval_libero.py`)

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
