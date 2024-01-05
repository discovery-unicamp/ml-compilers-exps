#!/bin/bash

# TODO: improve cli args passing method
CSV=$1
REPEAT=$2
DATASET=$3
DTYPE=$4

echo "Running GPU baselines"

eval "$(conda shell.bash hook)"
conda deactivate

python src/run_baseline.py --baseline cupy --file $CSV --repeat $REPEAT --dataset $DATASET --dtype $DTYPE
