#!/bin/bash

# Script to run the FlowerVLA Docker container

set -e

echo "=========================================="
echo "Running FlowerVLA Docker Container"
echo "=========================================="

# Create necessary directories if they don't exist
mkdir -p dataset checkpoints outputs logs

# Run the container with GPU support
docker run -it --rm \
    --gpus all \
    --shm-size=16g \
    --network host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e MUJOCO_GL=egl \
    -e PYOPENGL_PLATFORM=egl \
    -v $(pwd)/flower:/workspace/flower_vla_calvin/flower \
    -v $(pwd)/configs:/workspace/flower_vla_calvin/configs \
    -v $(pwd)/dataset:/workspace/flower_vla_calvin/dataset \
    -v $(pwd)/checkpoints:/workspace/flower_vla_calvin/checkpoints \
    -v $(pwd)/outputs:/workspace/flower_vla_calvin/outputs \
    -v $(pwd)/logs:/workspace/flower_vla_calvin/logs \
    -v $(pwd)/run_evaluation.sh:/workspace/flower_vla_calvin/run_evaluation.sh \
    -v $(pwd)/download_pretrained.sh:/workspace/flower_vla_calvin/download_pretrained.sh \
    -v $(pwd)/QUICKSTART.md:/workspace/flower_vla_calvin/QUICKSTART.md \
    -v ~/.cache/wandb:/root/.cache/wandb \
    -w /workspace/flower_vla_calvin \
    flower_vla_calvin:latest \
    /bin/bash

echo ""
echo "Container exited."
