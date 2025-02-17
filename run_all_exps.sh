#!/bin/bash

MAX_THREADS=
TAG_BASE="New Experiments"


python src/validate_jax.py --cpu --gpu --file validate_jax.txt
python src/validate_torch.py --cpu --gpu --file validate_torch.txt
# "64-64-256" "64-64-512" "128-128-512" "128-128-1024"
for exp in "64-64-256" "64-64-512" "128-128-512" "128-128-1024"; do
    echo "${exp} exps"
    python generate_experiment_scope.py --file experiments/index.json --cpu --gpu --size $exp --dtype float64 --tag "${TAG_BASE} Complex Trace Float 64 Shape ${exp}"

    python build_modules_on_scope.py --cpu 2 --gpu --scheduler ansor --ansor 1
    python src/validate_tvm.py --cpu --gpu --file "validate_tvm_1000_${exp}_f64.txt"

    python build_modules_on_scope.py --cpu 3 --scheduler ansor --ansor 1
    python src/validate_tvm.py --cpu --gpu --file "validate_tvm_1000_${exp}_f64_default.txt"
    # python run_experiment_on_scope.py --cpu --baseline --tvm 1 --jax --sample 1 --repeat 5 --max-threads $MAX_THREADS
done

# "32-32-32" "64-64-64" "128-128-128" 
for exp in "32-32-32" "64-64-64" "128-128-128"; do
    echo "${exp} exps"
    python generate_experiment_scope.py --file experiments/index.json --cpu --size $exp --dtype float32 --tag "${TAG_BASE} Conv/GLCM Float 32 Shape ${exp}"

    python build_modules_on_scope.py --cpu 2 --gpu --scheduler ansor --ansor 1
    python src/validate_tvm.py --cpu --file "validate_tvm_1000_${exp}_f32.txt"
    python build_modules_on_scope.py --cpu 3 --scheduler ansor --ansor 1

    python src/validate_tvm.py --cpu --file "validate_tvm_1000_${exp}_f32_default.txt"
    # python run_experiment_on_scope.py --cpu --baseline --tvm 1 --jax --sample 1 --repeat 5 --max-threads $MAX_THREADS
done

for exp in "32-32-32" "64-64-64" "128-128-128"; do
    echo "${exp} exps"
    python generate_experiment_scope.py --file experiments/index.json --cpu --size $exp --dtype float64 --tag "${TAG_BASE} Conv/GLCM Float 64 Shape ${exp}"


    python build_modules_on_scope.py --cpu 2 --scheduler ansor --ansor 1
    python src/validate_tvm.py --cpu --gpu --file "validate_tvm_1000_${exp}_f64.txt"

    python build_modules_on_scope.py --cpu 3 --scheduler ansor --ansor 1
    python src/validate_tvm.py --cpu --file "validate_tvm_1000_${exp}_f64_default.txt"
    # python run_experiment_on_scope.py --cpu --baseline --tvm 1 --jax --sample 1 --repeat 5 --max-threads $MAX_THREADS
done
