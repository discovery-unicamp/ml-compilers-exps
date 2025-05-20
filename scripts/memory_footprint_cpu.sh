#!/bin/bash

# TODO: improve cli args passing method
CSV=$1
ATTR=$2
RUNTIME=$3
DTYPE=$4
MAX_THREADS=$5
BASE_PATH=$6
DATASET_ID=$7
DATASET=$8
MEASURE_TYPE=$9



export OMP_NUM_THREADS=$MAX_THREADS
echo "OMP_NUM_THREADS set to ${OMP_NUM_THREADS}"

python -u src/memory_footprint_cpu_${MEASURE_TYPE}.py --runtime $RUNTIME --attribute $ATTR --base-path $BASE_PATH --file $CSV --dataset $DATASET --dataset-id $DATASET_ID --dtype $DTYPE
