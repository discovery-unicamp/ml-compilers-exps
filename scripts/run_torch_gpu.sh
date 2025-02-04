#!/bin/bash

# TODO: improve cli args passing method
CSV=$1
REPEAT=$2
DATASET=$3
DTYPE=$4

echo "Running GPU Torch operators"

eval "$(conda shell.bash hook)"
conda deactivate
python src/run_torch.py --arch gpu --file $CSV --repeat $REPEAT --dataset $DATASET --dtype $DTYPE
