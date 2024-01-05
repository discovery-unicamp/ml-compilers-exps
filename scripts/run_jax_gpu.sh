#!/bin/bash

# TODO: improve cli args passing method
CSV=$1
REPEAT=$2
DATASET=$3
DTYPE=$4

echo "Running GPU JAX operators"

eval "$(conda shell.bash hook)"
conda deactivate
export JAX_ENABLE_X64=true
python src/run_jax.py --arch gpu --file $CSV --repeat $REPEAT --dataset $DATASET --dtype $DTYPE
