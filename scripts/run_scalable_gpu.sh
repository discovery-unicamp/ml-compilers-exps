#!/bin/bash

# TODO: improve cli args passing method
CSV=$1
ATTR=$2
RUNTIME=$3
SHAPE=$4
DTYPE=$5
MAX_THREADS=$6
BASE_PATH=$7
ID=$8
INPUT_PATH=$9
OUTPUT_PATH=${10}



python -u src/run_scalable.py --arch gpu --runtime $RUNTIME --attribute $ATTR --base-path $BASE_PATH  --id $ID --file $CSV --input-path $INPUT_PATH --output-path $OUTPUT_PATH --shape $SHAPE --dtype $DTYPE
