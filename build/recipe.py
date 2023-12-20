python_version = USERARG.get("python-version", "3.10.13")
tvm_version = USERARG.get("tvm-version", "0.14.dev16")
numpy_version = USERARG.get("numpy-version", "1.24.3")
scipy_version = USERARG.get("scipy-version", "1.10.1")
base_image = USERARG.get("base-image", "tvm:base")

Stage0 += baseimage(image=base_image)

Stage0 += raw(
    docker='SHELL ["/bin/bash", "--login", "-c"]', singularity="# no equivalent"
)

Stage0 += apt_get(ospackages=["git", "pkg-config", "gfortran"])


Stage0 += comment("default env")
Stage0 += shell(
    commands=[
        f"conda create -n default python={python_version}",
        "conda activate default",
        f'''pip install scipy=={scipy_version} numpy=={numpy_version} jupyter apache-tvm=={tvm_version} \
                        dacite dask dask_cuda dask_jobqueue dask_memusage dask_ml dask-pytorch-ddp GPUtil \
                        gdown graphviz h5py hdbscan ipympl matplotlib memray networkx ormsgpack packaging \
                        portalocker psutil pyarrow pytorch-lightning scikit-learn torchvision xarray \
                        xgboost zarr glcm-cupy multidimio scikit-image segyio segysak py-cpuinfo \
                        bokeh==2.4.3 \"protobuf<=3.20.1\" \"charset-normalizer<3.0\" \"tornado<6.2\"''',
        "pip install --extra-index-url https://test.pypi.org/simple/ XPySom-dask"
    ]
)

Stage0 += shell(
    commands=[
        f"git clone --depth 1 --branch v{numpy_version} https://github.com/numpy/numpy.git",
        "cd numpy",
        "git submodule update --init",
        "cd ..",
        f"git clone --depth 1 --branch v{scipy_version} https://github.com/scipy/scipy.git",
        "cd scipy",
        "git submodule update --init",
    ]
)
Stage0 += comment("OpenBLAS env")
Stage0 += shell(
    commands=[
        f"conda create -n openblas python={python_version}",
        "conda activate openblas",
        "cd numpy",
        "echo -e '[openblas]\\nlibraries = openblas\\nlibrary_dirs = /opt/openblas/lib\\ninclude_dirs = /opt/openblas/include\\nruntime_library_dirs = /opt/openblas/lib\\nextra_link_args = -lm -fopenmp' > site.cfg",
        "pip install 'Cython<3.0.0'",
        "NPY_BLAS_ORDER=openblas NPY_LAPACK_ORDER=openblas python setup.py build -j 4 install",
        "rm -r build",
        "rm site.cfg",
        f'''pip install scipy=={scipy_version} numpy=={numpy_version} jupyter apache-tvm=={tvm_version} \
                        dacite dask dask_jobqueue dask_memusage dask_ml dask-pytorch-ddp GPUtil \
                        gdown graphviz h5py hdbscan ipympl matplotlib memray networkx ormsgpack packaging \
                        portalocker psutil pyarrow pytorch-lightning scikit-learn torchvision xarray \
                        xgboost zarr glcm-cupy multidimio scikit-image segyio segysak py-cpuinfo \
                        bokeh==2.4.3 \"protobuf<=3.20.1\" \"charset-normalizer<3.0\" \"tornado<6.2\"''',
        "pip install --extra-index-url https://test.pypi.org/simple/ XPySom-dask"
    ]
)

Stage0 += comment("AOCL env")
Stage0 += shell(
    commands=[
        f"conda create -n amd python={python_version}",
        "conda activate amd",
        "cd numpy",
        "echo -e '[blis]\\nlibraries = blis\\nlibrary_dirs = /opt/amd/amd-blis/lib/ILP64\\ninclude_dirs = /opt/amd/amd-blis/include/ILP64\\nruntime_library_dirs = /opt/amd/amd-blis/lib/ILP64\\nextra_link_args = -lm -fopenmp' > site.cfg",
        "echo -e '[flame]\\nlibraries = flame\\nlibrary_dirs = /opt/amd/amd-libflame/lib/ILP64\\ninclude_dirs = /opt/amd/amd-libflame/include/ILP64\\nruntime_library_dirs = /opt/amd/amd-libflame/lib/ILP64\\nextra_link_args = -lm -fopenmp' >> site.cfg",
        "pip install 'Cython<3.0.0'",
        "NPY_BLAS_ORDER=blis NPY_LAPACK_ORDER=flame python setup.py build -j 4 install",
        "rm -r build",
        "rm site.cfg",
        f'''pip install scipy=={scipy_version} numpy=={numpy_version} jupyter apache-tvm=={tvm_version} \
                        dacite dask dask_cuda dask_jobqueue dask_memusage dask_ml dask-pytorch-ddp GPUtil \
                        gdown graphviz h5py hdbscan ipympl matplotlib memray networkx ormsgpack packaging \
                        portalocker psutil pyarrow pytorch-lightning scikit-learn torchvision xarray \
                        xgboost zarr glcm-cupy multidimio scikit-image segyio segysak py-cpuinfo \
                        bokeh==2.4.3 \"protobuf<=3.20.1\" \"charset-normalizer<3.0\" \"tornado<6.2\"''',
        "pip install --extra-index-url https://test.pypi.org/simple/ XPySom-dask"
    ]
)

Stage0 += comment("AOCL GCC env")
Stage0 += shell(
    commands=[
        f"conda create -n amd_gcc python={python_version}",
        "conda activate amd_gcc",
        "cd numpy",
        "echo -e '[blis]\\nlibraries = blis\\nlibrary_dirs = /opt/amd_gcc/amd-blis/lib/ILP64\\ninclude_dirs = /opt/amd_gcc/amd-blis/include/ILP64\\nruntime_library_dirs = /opt/amd_gcc/amd-blis/lib/ILP64\\nextra_link_args = -lm -fopenmp -lrt -lpthread' > site.cfg",
        "echo -e '[flame]\\nlibraries = flame\\nlibrary_dirs = /opt/amd_gcc/amd-libflame/lib/ILP64\\ninclude_dirs = /opt/amd_gcc/amd-libflame/include/ILP64\\nruntime_library_dirs = /opt/amd_gcc/amd-libflame/lib/ILP64\\nextra_link_args = -lm -fopenmp -lgfortran -lquadmath' >> site.cfg",
        "pip install 'Cython<3.0.0'",
        "NPY_BLAS_ORDER=blis NPY_LAPACK_ORDER=flame python setup.py build -j 4 install",
        "rm -r build",
        "rm site.cfg",
        f'''pip install scipy=={scipy_version} numpy=={numpy_version} jupyter apache-tvm=={tvm_version} \
                        dacite dask dask_cuda dask_jobqueue dask_memusage dask_ml dask-pytorch-ddp GPUtil \
                        gdown graphviz h5py hdbscan ipympl matplotlib memray networkx ormsgpack packaging \
                        portalocker psutil pyarrow pytorch-lightning scikit-learn torchvision xarray \
                        xgboost zarr glcm-cupy multidimio scikit-image segyio segysak py-cpuinfo \
                        bokeh==2.4.3 \"protobuf<=3.20.1\" \"charset-normalizer<3.0\" \"tornado<6.2\"''',
        "pip install --extra-index-url https://test.pypi.org/simple/ XPySom-dask"
    ]
)

Stage0 += comment("Intel MKL Conda env")
Stage0 += shell(
    commands=[
        "conda config --add channels intel",
        f"conda create -n intel_conda intelpython3_core python={python_version}",
        "conda activate intel_conda",
        f'''pip install scipy=={scipy_version} numpy=={numpy_version} jupyter apache-tvm=={tvm_version} \
                        dacite dask dask_jobqueue dask_memusage dask_ml dask-pytorch-ddp GPUtil \
                        gdown graphviz h5py hdbscan ipympl matplotlib memray networkx ormsgpack packaging \
                        portalocker psutil pyarrow pytorch-lightning scikit-learn torchvision xarray \
                        xgboost zarr glcm-cupy multidimio scikit-image segyio segysak py-cpuinfo \
                        bokeh==2.4.3 \"protobuf<=3.20.1\" \"charset-normalizer<3.0\" \"tornado<6.2\"''',
        "pip install --extra-index-url https://test.pypi.org/simple/ XPySom-dask"
    ]
)
