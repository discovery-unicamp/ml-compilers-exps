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

nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=cpu \
  --output=mem_trace \
  --export=sqlite \
  python -u src/memory_footprint_gpu.py --runtime "baseline_gpu" --attribute "envelope" --base-path "/home/joao.serodio/native-tvm/experiments/modules_1/1bf036c5-f567-44c6-a22a-5db525d96635" --file "/home/joao.serodio/native-tvm/experiments/results_1/1bf036c5-f567-44c6-a22a-5db525d96635/Memory.csv" --dataset "/home/joao.serodio/native-tvm/data/parihaka/128-128-512/1.npy" --dataset-id 1 --dtype "float64" --nsys