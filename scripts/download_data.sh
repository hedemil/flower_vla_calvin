#!/bin/bash

# Download CALVIN datasets into the dataset directory
# Usage: ./download_data.sh [D|ABC|ABCD|debug]

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${SCRIPT_DIR}/dataset"

# Create dataset directory if it doesn't exist
mkdir -p "${DATASET_DIR}"

# Change to dataset directory
cd "${DATASET_DIR}"

# Download, Unzip, and Remove zip
if [ "$1" = "D" ]
then

    echo "Downloading task_D_D to ${DATASET_DIR}..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
    unzip task_D_D.zip
    rm task_D_D.zip
    echo "saved folder: ${DATASET_DIR}/task_D_D"
elif [ "$1" = "ABC" ]
then

    echo "Downloading task_ABC_D to ${DATASET_DIR}..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
    unzip task_ABC_D.zip
    rm task_ABC_D.zip
    echo "saved folder: ${DATASET_DIR}/task_ABC_D"

elif [ "$1" = "ABCD" ]
then

    echo "Downloading task_ABCD_D to ${DATASET_DIR}..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip
    unzip task_ABCD_D.zip
    rm task_ABCD_D.zip
    echo "saved folder: ${DATASET_DIR}/task_ABCD_D"

elif [ "$1" = "debug" ]
then

    echo "Downloading debug dataset to ${DATASET_DIR}..."
    wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
    unzip calvin_debug_dataset.zip
    rm calvin_debug_dataset.zip
    echo "saved folder: ${DATASET_DIR}/calvin_debug_dataset"


else
    echo "Failed: Usage download_data.sh D | ABC | ABCD | debug"
    exit 1
fi
