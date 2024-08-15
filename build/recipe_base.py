tvm_version = USERARG.get("tvm-version", "dev0.14.2")
jax_version = USERARG.get("jax-version", "0.4.23")
numpy_version = USERARG.get("numpy-version", "1.24.3")
scipy_version = USERARG.get("scipy-version", "1.10.1")
openblas_target = USERARG.get("openblas-target", "SKYLAKEX")
llvm_target = USERARG.get("llvm-target", "x86")
cmake = USERARG.get("cmake", "config.cmake")


llvm_options = {
    "x86": "https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz",
    "armv7a": "https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-armv7a-linux-gnueabihf.tar.xz",
    "aarch64": "https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-aarch64-linux-gnu.tar.xz",
}

Stage0 += baseimage(image="nvcr.io/nvidia/pytorch:23.08-py3")

Stage0 += raw(
    docker='SHELL ["/bin/bash", "--login", "-c"]', singularity="# no equivalent"
)

Stage0 += apt_get(ospackages=["python3-dev", "python3-setuptools", "gcc", "libtinfo-dev", "zlib1g-dev", "build-essential", "cmake", "libedit-dev", "libxml2-dev"])


Stage0 += shell(
    commands=[
        f"git clone --branch v1.5.5 https://github.com/facebook/zstd.git",
        "cd zstd",
        "cmake build/cmake",
        "make -j 4",
        "make install",
        "cd ..",
        "rm -r zstd",
    ]
)

Stage0 += shell(
    commands=[
        f"wget {llvm_options[llvm_target]} -O llvm.tar.xz",
        "mkdir /opt/llvm",
        "tar xf llvm.tar.xz --strip-components=1 --directory /opt/llvm",
        "rm llvm.tar.xz"
    ]
)


Stage0 += shell(
    commands=[
        "pip install -U --pre cupy-cuda12x -f https://pip.cupy.dev/pre",
        f'''pip install scipy=={scipy_version} numpy=={numpy_version} jupyter \
                        dacite==1.8.1 dask==2023.3.2 dask_cuda==23.6.0 dask_jobqueue==0.8.2 dask_memusage==1.1 GPUtil==1.4.0 \
                        gdown==4.7.1 graphviz==0.20.1 h5py==3.10.0 ipympl==0.9.3 matplotlib==3.7.2 memray==1.11.0 networkx==3.2.1 ormsgpack==1.4.1 packaging==23.1 \
                        portalocker==2.8.2 psutil==5.9.4 pyarrow==11.0.0 xarray==2023.12.0 scikit-image==0.22.0 \
                        xgboost==1.7.5 zarr==2.16.1 glcm-cupy==0.2.1 multidimio==0.4.2 segyio==1.9.12 segysak==0.3.4 py-cpuinfo==9.0.0 \
                        bokeh==2.4.3 protobuf==3.20.1 charset-normalizer==2.1.1 tornado==6.1''',
        f"pip install -U \"jax[cuda12_pip]=={jax_version}\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    ]
)

Stage0 += copy(src=cmake, dest="/tmp/config.cmake")
Stage0 += shell(
    commands=[
        f"git clone --recursive --branch {tvm_version} https://github.com/SerodioJ/tvm",
        "cd tvm",
        "mkdir build",
        "cp /tmp/config.cmake build",
        "cd build",
        "cmake ..",
        "make -j 4",
        "make install",
        "cd ..",
        "cd python",
        "python setup.py install",
        "cd ../..",
        # "rm -r tvm"
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
