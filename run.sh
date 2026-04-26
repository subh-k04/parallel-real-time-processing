#!/bin/bash
set -e

if [ "$1" == "cpu" ]; then
    echo "Starting CPU mode..."
    mpirun -np 4 ./blur_cpu
    echo "CPU mode finished."
elif [ "$1" == "gpu" ]; then
    echo "Starting GPU mode..."
    mpirun -np 2 ./blur_gpu
    echo "GPU mode finished."
else
    echo "Usage: ./run.sh [cpu|gpu]"
    exit 1
fi
