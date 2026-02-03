#!/bin/bash

# Script to run the FlowerVLA Docker container

set -e

echo "=========================================="
echo "Running FlowerVLA Docker Container"
echo "=========================================="

# Create necessary directories if they don't exist
mkdir -p dataset checkpoints outputs logs ~/.libero

# Run the container with GPU support
docker run -it --rm \
    --gpus all \
    --shm-size=16g \
    --network host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e MUJOCO_GL=egl \
    -e PYOPENGL_PLATFORM=egl \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -v $(pwd)/conf:/workspace/flower_vla_calvin/conf \
    -v $(pwd)/flower:/workspace/flower_vla_calvin/flower \
    -v $(pwd)/calvin_env:/workspace/flower_vla_calvin/calvin_env \
    -v $(pwd)/configs:/workspace/flower_vla_calvin/configs \
    -v $(pwd)/dataset:/workspace/flower_vla_calvin/dataset \
    -v $(pwd)/checkpoints:/workspace/flower_vla_calvin/checkpoints \
    -v $(pwd)/outputs:/workspace/flower_vla_calvin/outputs \
    -v $(pwd)/LIBERO/libero/datasets:/workspace/flower_vla_calvin/LIBERO/libero/datasets \
    -v $(pwd)/logs:/workspace/flower_vla_calvin/logs \
    -v $(pwd)/preprocess:/workspace/flower_vla_calvin/preprocess \
    -v $(pwd)/scripts/run_evaluation.sh:/workspace/flower_vla_calvin/run_evaluation.sh \
    -v $(pwd)/scripts/run_libero_evaluation.sh:/workspace/flower_vla_calvin/run_libero_evaluation.sh \
    -v $(pwd)/scripts/run_training.sh:/workspace/flower_vla_calvin/run_training.sh \
    -v $(pwd)/scripts/run_training_cluster.sh:/workspace/flower_vla_calvin/run_training_cluster.sh \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/wandb:/root/.cache/wandb \
    -w /workspace/flower_vla_calvin \
    flower_vla_calvin:latest \
    /bin/bash

echo ""
echo "Container exited."
