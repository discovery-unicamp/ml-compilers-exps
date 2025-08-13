#!/bin/bash

# TODO: improve cli args passing method
CSV=$1
REPEAT=$2
DATASET=$3
DTYPE=$4
BASE_PATH=$5
ID=$6

echo "Running GPU TVM operators"

# eval "$(conda shell.bash hook)"
# conda deactivate
python src/run_tvm.py --arch gpu --base-path $BASE_PATH --id $ID --file $CSV --repeat $REPEAT --dataset $DATASET --dtype $DTYPE
