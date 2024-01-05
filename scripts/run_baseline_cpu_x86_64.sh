#!/bin/bash

# TODO: improve cli args passing method
CSV=$1
REPEAT=$2
DATASET=$3
DTYPE=$4
MAX_THREADS=$5

echo "CPU baselines"
eval "$(conda shell.bash hook)"

for env in  default intel_conda amd amd_gcc openblas; do
    echo "Running on ${env}"
    conda deactivate
    conda activate $env
    export OMP_NUM_THREADS=$MAX_THREADS
    echo "OMP_NUM_THREADS set to ${OMP_NUM_THREADS}"
    python src/run_baseline.py --baseline $env --file $CSV --repeat $REPEAT --dataset $DATASET --dtype $DTYPE
done
