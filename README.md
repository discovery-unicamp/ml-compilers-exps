# Using ML compilers for High Performance Seismic Attribute Computation

## Configuring the environment

First install DVC and HPCCM via pip, utilities that will be required to setup the current environment:

```
pip install "dvc[s3]==3.42.0" hpccm
```

## Acessing and updating DVC storage

DVC is a tool for versioning data taht is not suitable to be stored on a GIT repository
To get data from remote storage just run:

```
dvc pull
```

New data can be added using `dvc add` and pushed using `dvc push`.

## Building Docker image

On the build folder, there are two HPCCM recipes that can be used to generate Dockerfiles

```
hpccm --recipe recipe_base.py  --userarg openblas-target=ZEN > Dockerfile.base
```

```
hpccm --recipe recipe.py  --userarg base-image=tvm:base_zen > Dockerfile
```

To build the images:

```
docker build -t tvm:base_zen -f Dockerfile.base .
```

```
docker build -t tvm:zen -f Dockerfile .
```

## Instructions on how to run experiments

### Creating an experiment scope

When at least one of the items below change, a new experiment scope should be created.

- machine being used
- input data shape
- input data dtype
- git hash (when relevant changes are made to scripts, Docker image or operator source code)

On a Docker container from a previously created image after deactivating the conda environment, run

```
python generate_experiment_scope.py --file experiments/experiment_index.json --cpu --gpu --size 64-64-512 --dtype float64 --tag "Tag describing experiment"
```

This will add a new key:value pair at the end of the JSON file, with information about the experiment and the machine used to run the experiment. The modules built and run results will be stored on experiments/modules/<EXP_ID> and experiments/results/<EXP_ID>, respectively.

### Building modules from Tensor Expression code

When building TVM modules, 3 different schedulers can be used, default, ansor and autotvm (AutoTVM is not supported by the scripts yet), and some parameters when building are extracted from the a [Build Profiles JSON file](experiments/build_profiles.json). New profiles can be added in their respective field (cpu, gpu or ansor), but at the end of the list, without changing the current position of previously set profiles.

On a Docker container from a previously created image after deactivating the conda environment, run

```
python build_modules_on_scope.py --cpu 1 --gpu 0 --scheduler ansor --ansor 1
```

The flags `cpu`, `gpu` and `ansor` select the profiles from their respective lists on the build profiles JSON file. By default the Experiment ID selected will be the last on the Index JSON file, but a different one can be chosen using the flag `experiment-id`.

For each build a new folder is created inside the experiment directory, following the name pattern `Build<BUILD_ID>` in ascending order.

### Validating the operators

To validate JAX operators just run

```
python src/validate_jax.py --cpu --gpu --file jax.txt
```

It will create `jax.txt` file containing the error statistics for each dataset and flot precision, in a 4-tuple - MEAN_ABS_ERR|STD_ABS_ERR|MAX_ABS_ERR|MAX_ABS_REL_ERR.

For TVM operator, it is similar but it has 2 addtional flags that can be used to select the Experiment ID (`experiment-id`) and Build ID (`build-id`).

```
python src/validate_tvm.py --cpu --gpu --file tvm.txt
```

### Running performance experiments

Performance experiments are run within the scope of the experiment, so it will use only the shape set when creating a scope.

```
python run_experiment_on_scope.py --cpu --gpu --baseline --tvm 1 --jax --sample 1 --repeat 5 --max-threads 6
```

The CLI command presented above runs the CPU (`--cpu`) and GPU (`--gpu`) operatos, for the baselines (`--baseline`), JAX (`--jax`) and TVM operators from Build 1 of the current scope (`--tvm 1`). It will use the sample 1 from the dataset, repeat the measuments 5 times and set OMP_NUM_THREADS to 6. A different Experiment Id can be selected using the `experiment-id` flag.


### Running memory profiling experiments

Only MemRay and NsightSystems are installed in the base image, to run `mprof` and `/usr/bin/time` experiments install them with:

```
pip install memory_profiler
apt install time
```

## Third Party software
This project includes code based on the following projects:
- [DASF Seismic](https://github.com/discovery-unicamp/dasf-seismic): MIT License
- [d2geo](https://github.com/dudley-fitzgerald/d2geo): GPL-3.0 license
