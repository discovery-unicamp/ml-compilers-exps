tvm_version = USERARG.get("tvm-version", "dev0.19.1")
openblas_target = USERARG.get("openblas-target", "SKYLAKEX")
llvm_target = USERARG.get("llvm-target", "x86")
cmake = USERARG.get("cmake", "config.cmake")
requirements = USERARG.get("requirements", "requirements.txt")


llvm_options = {
    "x86": "https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz",
    "armv7a": "https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-armv7a-linux-gnueabihf.tar.xz",
    "aarch64": "https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-aarch64-linux-gnu.tar.xz",
}

Stage0 += baseimage(image="nvcr.io/nvidia/pytorch:24.05-py3")

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

Stage0 += copy(src=requirements, dest="/tmp/requirements.txt")
Stage0 += shell(
    commands=[
        "pip install -r /tmp/requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
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
        "rm -r tvm"
    ]
)

# Stage0 += openblas(
#     prefix="/opt/openblas", make_opts=[f"TARGET={openblas_target}", "USE_OPENMP=1"]
# )
# Stage0 += copy(
#     src=[
#         "aocl/aocl-blis-linux-aocc-4.0.tar.gz",
#         "aocl/aocl-libflame-linux-aocc-4.0.tar.gz",
#         "aocl/aocl-blis-linux-gcc-4.0.tar.gz",
#         "aocl/aocl-libflame-linux-gcc-4.0.tar.gz",
#     ],
#     dest="/tmp",
# )
# Stage0 += shell(
#     commands=[
#         "mkdir /opt/amd",
#         "tar -xf /tmp/aocl-blis-linux-aocc-4.0.tar.gz -C /opt/amd",
#         "tar -xf /tmp/aocl-libflame-linux-aocc-4.0.tar.gz -C /opt/amd",
#         "mkdir /opt/amd_gcc",
#         "tar -xf /tmp/aocl-blis-linux-gcc-4.0.tar.gz -C /opt/amd_gcc",
#         "tar -xf /tmp/aocl-libflame-linux-gcc-4.0.tar.gz -C /opt/amd_gcc",
#     ]
# )
# Stage0 += conda(eula=True)
