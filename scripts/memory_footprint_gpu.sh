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
NSYS=$9

if [[ "$NSYS" == "true" ]]; then
  rm  gpu_mem_trace.nsys-rep
  rm  gpu_mem_trace.sqlite
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --sample=cpu \
    --output=gpu_mem_trace \
    --export=sqlite \
    python -u src/memory_footprint_gpu.py --runtime $RUNTIME --attribute $ATTR --base-path $BASE_PATH --file $CSV --dataset $DATASET --dataset-id $DATASET_ID --dtype $DTYPE --nsys
else
  python -u src/memory_footprint_gpu.py --runtime $RUNTIME --attribute $ATTR --base-path $BASE_PATH --file $CSV --dataset $DATASET --dataset-id $DATASET_ID --dtype $DTYPE
fi



# nsys python -u src/memory_footprint_gpu.py --runtime $RUNTIME --attribute $ATTR --base-path $BASE_PATH  --id $ID --file $CSV --input-path $INPUT_PATH --output-path $OUTPUT_PATH --shape $SHAPE --dtype $DTYPE

