#!/bin/bash

# Script to build the FlowerVLA Docker image

set -e

echo "=========================================="
echo "Building FlowerVLA Docker Image"
echo "=========================================="

# Build the Docker image
docker build -t flower_vla_calvin:latest .

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "To run the container, use:"
echo "  ./docker_run.sh"
echo ""
echo "Or with docker-compose:"
echo "  docker-compose up -d"
echo "  docker-compose exec flower_vla bash"
echo ""
