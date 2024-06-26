#!/bin/bash

# TODO: improve cli args passing method
CSV=$1
REPEAT=$2
DATASET=$3
DTYPE=$4
MAX_THREADS=$5

echo "Running CPU JAX operators"

eval "$(conda shell.bash hook)"
conda deactivate
export JAX_ENABLE_X64=true

export OMP_NUM_THREADS=$MAX_THREADS
echo "OMP_NUM_THREADS set to ${OMP_NUM_THREADS}"
python src/run_jax.py --arch cpu --file $CSV --repeat $REPEAT --dataset $DATASET --dtype $DTYPE
