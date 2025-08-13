python_version = USERARG.get("python-version", "3.10.13")
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
        f'''pip install scipy=={scipy_version} numpy=={numpy_version} jupyter \
                        dacite==1.8.1 dask==2023.3.2 dask_cuda==23.6.0 dask_jobqueue==0.8.2 dask_memusage==1.1 GPUtil==1.4.0 \
                        gdown==4.7.1 graphviz==0.20.1 h5py==3.10.0 ipympl==0.9.3 matplotlib==3.7.2 memray==1.11.0 networkx==3.2.1 ormsgpack==1.4.1 packaging==23.1 \
                        portalocker==2.8.2 psutil==5.9.4 pyarrow==11.0.0 xarray==2023.12.0 scikit-image==0.22.0 \
                        xgboost==1.7.5 zarr==2.16.1 glcm-cupy==0.2.1 multidimio==0.4.2 segyio==1.9.12 segysak==0.3.4 py-cpuinfo==9.0.0 \
                        bokeh==2.4.3 protobuf==3.20.1 charset-normalizer==2.1.1 tornado==6.1''',
    ]
)

Stage0 += shell(
    commands=[
        f"git clone --depth 1 --branch v{numpy_version} https://github.com/numpy/numpy.git",
        "cd numpy",
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
        f'''pip install scipy=={scipy_version} numpy=={numpy_version} jupyter \
                        dacite==1.8.1 dask==2023.3.2 dask_cuda==23.6.0 dask_jobqueue==0.8.2 dask_memusage==1.1 GPUtil==1.4.0 \
                        gdown==4.7.1 graphviz==0.20.1 h5py==3.10.0 ipympl==0.9.3 matplotlib==3.7.2 memray==1.11.0 networkx==3.2.1 ormsgpack==1.4.1 packaging==23.1 \
                        portalocker==2.8.2 psutil==5.9.4 pyarrow==11.0.0 xarray==2023.12.0 scikit-image==0.22.0 \
                        xgboost==1.7.5 zarr==2.16.1 glcm-cupy==0.2.1 multidimio==0.4.2 segyio==1.9.12 segysak==0.3.4 py-cpuinfo==9.0.0 \
                        bokeh==2.4.3 protobuf==3.20.1 charset-normalizer==2.1.1 tornado==6.1''',
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
        f'''pip install scipy=={scipy_version} numpy=={numpy_version} jupyter \
                        dacite==1.8.1 dask==2023.3.2 dask_cuda==23.6.0 dask_jobqueue==0.8.2 dask_memusage==1.1 GPUtil==1.4.0 \
                        gdown==4.7.1 graphviz==0.20.1 h5py==3.10.0 ipympl==0.9.3 matplotlib==3.7.2 memray==1.11.0 networkx==3.2.1 ormsgpack==1.4.1 packaging==23.1 \
                        portalocker==2.8.2 psutil==5.9.4 pyarrow==11.0.0 xarray==2023.12.0 scikit-image==0.22.0 \
                        xgboost==1.7.5 zarr==2.16.1 glcm-cupy==0.2.1 multidimio==0.4.2 segyio==1.9.12 segysak==0.3.4 py-cpuinfo==9.0.0 \
                        bokeh==2.4.3 protobuf==3.20.1 charset-normalizer==2.1.1 tornado==6.1''',
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
        f'''pip install scipy=={scipy_version} numpy=={numpy_version} jupyter \
                        dacite==1.8.1 dask==2023.3.2 dask_cuda==23.6.0 dask_jobqueue==0.8.2 dask_memusage==1.1 GPUtil==1.4.0 \
                        gdown==4.7.1 graphviz==0.20.1 h5py==3.10.0 ipympl==0.9.3 matplotlib==3.7.2 memray==1.11.0 networkx==3.2.1 ormsgpack==1.4.1 packaging==23.1 \
                        portalocker==2.8.2 psutil==5.9.4 pyarrow==11.0.0 xarray==2023.12.0 scikit-image==0.22.0 \
                        xgboost==1.7.5 zarr==2.16.1 glcm-cupy==0.2.1 multidimio==0.4.2 segyio==1.9.12 segysak==0.3.4 py-cpuinfo==9.0.0 \
                        bokeh==2.4.3 protobuf==3.20.1 charset-normalizer==2.1.1 tornado==6.1''',
    ]
)

## Channel is broken
# Stage0 += comment("Intel MKL Conda env")
# Stage0 += shell(
#     commands=[
#         "conda config --add channels intel",
#         f"conda create -n intel_conda intelpython3_core python={python_version}",
#         "conda activate intel_conda",
#         f'''pip install scipy=={scipy_version} numpy=={numpy_version} jupyter \
#                         dacite==1.8.1 dask==2023.3.2 dask_cuda==23.6.0 dask_jobqueue==0.8.2 dask_memusage==1.1 GPUtil==1.4.0 \
#                         gdown==4.7.1 graphviz==0.20.1 h5py==3.10.0 ipympl==0.9.3 matplotlib==3.7.2 memray==1.11.0 networkx==3.2.1 ormsgpack==1.4.1 packaging==23.1 \
#                         portalocker==2.8.2 psutil==5.9.4 pyarrow==11.0.0 xarray==2023.12.0 scikit-image==0.22.0 \
#                         xgboost==1.7.5 zarr==2.16.1 glcm-cupy==0.2.1 multidimio==0.4.2 segyio==1.9.12 segysak==0.3.4 py-cpuinfo==9.0.0 \
#                         bokeh==2.4.3 protobuf==3.20.1 charset-normalizer==2.1.1 tornado==6.1''',
#     ]
# )
