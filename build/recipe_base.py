tvm_version = USERARG.get("tvm-version", "0.14.dev16")
numpy_version = USERARG.get("numpy-version", "1.24.3")
scipy_version = USERARG.get("scipy-version", "1.10.1")
openblas_target = USERARG.get("openblas-target", "SKYLAKEX")

Stage0 += baseimage(image="nvcr.io/nvidia/pytorch:23.06-py3")

Stage0 += raw(
    docker='SHELL ["/bin/bash", "--login", "-c"]', singularity="# no equivalent"
)

Stage0 += apt_get(ospackages=["build-essential"])

Stage0 += shell(
    commands=[
        "pip install -U --pre cupy-cuda12x -f https://pip.cupy.dev/pre",
        f'''pip install scipy=={scipy_version} numpy=={numpy_version} jupyter apache-tvm=={tvm_version} \
                        dacite dask dask_cuda dask_jobqueue dask_memusage GPUtil \
                        gdown graphviz h5py ipympl matplotlib memray networkx ormsgpack packaging \
                        portalocker psutil pyarrow xarray scikit-image \
                        xgboost zarr glcm-cupy multidimio segyio segysak py-cpuinfo \
                        bokeh==2.4.3 \"protobuf<=3.20.1\" \"charset-normalizer<3.0\" \"tornado<6.2\"''',
    ]
)

Stage0 += openblas(
    prefix="/opt/openblas", make_opts=[f"TARGET={openblas_target}", "USE_OPENMP=1"]
)
Stage0 += copy(
    src=[
        "aocl/aocl-blis-linux-aocc-4.0.tar.gz",
        "aocl/aocl-libflame-linux-aocc-4.0.tar.gz",
        "aocl/aocl-blis-linux-gcc-4.0.tar.gz",
        "aocl/aocl-libflame-linux-gcc-4.0.tar.gz",
    ],
    dest="/tmp",
)
Stage0 += shell(
    commands=[
        "mkdir /opt/amd",
        "tar -xf /tmp/aocl-blis-linux-aocc-4.0.tar.gz -C /opt/amd",
        "tar -xf /tmp/aocl-libflame-linux-aocc-4.0.tar.gz -C /opt/amd",
        "mkdir /opt/amd_gcc",
        "tar -xf /tmp/aocl-blis-linux-gcc-4.0.tar.gz -C /opt/amd_gcc",
        "tar -xf /tmp/aocl-libflame-linux-gcc-4.0.tar.gz -C /opt/amd_gcc",
    ]
)
Stage0 += conda(eula=True)
