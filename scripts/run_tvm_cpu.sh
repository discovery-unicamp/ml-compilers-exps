#!/bin/bash

# TODO: improve cli args passing method
CSV=$1
REPEAT=$2
DATASET=$3
DTYPE=$4
MAX_THREADS=$5
BASE_PATH=$6
ID=$7

echo "Running CPU TVM operators"

# eval "$(conda shell.bash hook)"
# conda deactivate

export OMP_NUM_THREADS=$MAX_THREADS
echo "OMP_NUM_THREADS set to ${OMP_NUM_THREADS}"
python src/run_tvm.py --arch cpu --base-path $BASE_PATH  --id $ID --file $CSV --repeat $REPEAT --dataset $DATASET --dtype $DTYPE

