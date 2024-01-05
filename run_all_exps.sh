#!/bin/bash

MAX_THREADS=
TAG_BASE=
eval "$(conda shell.bash hook)"
conda deactivate

python src/validate_jax.py --cpu --gpu --file validate_jax.txt

for exp in "64-64-256" "64-64-512" "128-128-512" "128-128-1024"; do
    echo "${exp} exps"
    python generate_experiment_scope.py --file experiments/experiment_index.json --cpu --gpu --size $exp --dtype float64 --tag "${TAG_BASE} Shape ${exp}"

    python build_modules_on_scope.py --cpu 1 --gpu 0 --scheduler ansor --ansor 0
    python src/validate_tvm.py --cpu --gpu --file "validate_tvm_10_${exp}.txt"
    python run_experiment_on_scope.py --cpu --gpu --baseline --tvm 1 --jax --sample 1 --repeat 5 --max-threads $MAX_THREADS

    python build_modules_on_scope.py --cpu 1 --gpu 0 --scheduler ansor --ansor 1
    python src/validate_tvm.py --cpu --gpu --file "validate_tvm_1000_${exp}.txt"
    python run_experiment_on_scope.py --cpu --gpu --baseline --tvm 2 --jax --sample 1 --repeat 5 --max-threads $MAX_THREADS
done