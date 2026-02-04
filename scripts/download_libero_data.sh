#!/bin/bash

# Download LIBERO datasets
# Usage: ./download_libero_data.sh [libero_goal|libero_spatial|libero_object|libero_10|libero_90|all]

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
LIBERO_DIR="${PROJECT_ROOT}/LIBERO"

# Detect Python command (python3 or python)
if command -v python3 &> /dev/null; then
    PYTHON=python3
    PIP=pip3
elif command -v python &> /dev/null; then
    PYTHON=python
    PIP=pip
else
    echo "Error: Neither python nor python3 found in PATH"
    exit 1
fi

# Check if LIBERO directory exists
if [ ! -d "${LIBERO_DIR}" ]; then
    echo "Error: LIBERO directory not found at ${LIBERO_DIR}"
    echo "Please ensure you're running this from the correct project directory"
    exit 1
fi

# Change to LIBERO directory
cd "${LIBERO_DIR}"

# Install LIBERO package if needed
if ! $PYTHON -c "import libero" 2>/dev/null; then
    echo "Installing LIBERO package and dependencies..."
    $PIP install -e . --quiet
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install LIBERO package"
        exit 1
    fi
    echo "LIBERO package installed successfully"
fi

# Check for required dependencies
MISSING_DEPS=()
if ! $PYTHON -c "import tqdm" 2>/dev/null; then
    MISSING_DEPS+=("tqdm")
fi
if ! $PYTHON -c "import huggingface_hub" 2>/dev/null; then
    MISSING_DEPS+=("huggingface_hub")
fi
if ! $PYTHON -c "import termcolor" 2>/dev/null; then
    MISSING_DEPS+=("termcolor")
fi

# Install missing dependencies
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "Installing missing dependencies: ${MISSING_DEPS[*]}"
    $PIP install "${MISSING_DEPS[@]}" --quiet
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
    echo "Dependencies installed successfully"
fi

# Function to download a benchmark
download_benchmark() {
    local benchmark=$1
    echo "Downloading ${benchmark} to ${LIBERO_DIR}/LIBERO/${benchmark}..."
    $PYTHON benchmark_scripts/download_libero_datasets.py --datasets "${benchmark}" --use-huggingface
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded ${benchmark}"
        echo "  saved folder: ${LIBERO_DIR}/LIBERO/${benchmark}"
    else
        echo "✗ Failed to download ${benchmark}"
        return 1
    fi
}

# Parse arguments and download
if [ "$1" = "libero_goal" ]; then
    download_benchmark "libero_goal"

elif [ "$1" = "libero_spatial" ]; then
    download_benchmark "libero_spatial"

elif [ "$1" = "libero_object" ]; then
    download_benchmark "libero_object"

elif [ "$1" = "libero_10" ]; then
    download_benchmark "libero_10"

elif [ "$1" = "libero_90" ]; then
    download_benchmark "libero_90"

elif [ "$1" = "all" ]; then
    echo "Downloading all LIBERO benchmarks..."
    echo ""
    download_benchmark "libero_goal"
    echo ""
    download_benchmark "libero_spatial"
    echo ""
    download_benchmark "libero_object"
    echo ""
    download_benchmark "libero_10"
    echo ""
    download_benchmark "libero_90"
    echo ""
    echo "All LIBERO benchmarks downloaded"

else
    echo "Failed: Usage download_libero_data.sh libero_goal | libero_spatial | libero_object | libero_10 | libero_90 | all"
    echo ""
    echo "Available benchmarks:"
    echo "  libero_goal    - Goal-conditioned tasks"
    echo "  libero_spatial - Spatial reasoning tasks"
    echo "  libero_object  - Object manipulation tasks"
    echo "  libero_10      - 10 diverse tasks"
    echo "  libero_90      - 90 diverse tasks"
    echo "  all            - Download all benchmarks"
    exit 1
fi
