python_version = USERARG.get("python-version", "3.10.8")
tvm_version = USERARG.get("tvm-version", "0.14.dev16")

Stage0 += baseimage(image="tvm:base")

Stage0 += raw(
    docker='SHELL ["/bin/bash", "--login", "-c"]', singularity="# no equivalent"
)

Stage0 += apt_get(ospackages=["git", "pkg-config", "gfortran"])


Stage0 += comment("default env")
Stage0 += shell(
    commands=[
        f"conda create -n default python={python_version}",
        "conda activate default",
        "pip install scipy==1.7.3 jupyter",
        "pip install numpy==1.23.5",
        f"pip install --no-deps apache-tvm=={tvm_version} cloudpickle==2.2.1 ml-dtypes==0.2",
    ]
)

Stage0 += shell(
    commands=[
        "git clone --depth 1 --branch v1.23.5 https://github.com/numpy/numpy.git",
        "cd numpy",
        "git submodule update --init",
        "cd ..",
        "git clone --depth 1 --branch v1.7.3 https://github.com/scipy/scipy.git",
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
        "pip install jupyter",
        f"pip install --no-deps apache-tvm=={tvm_version} cloudpickle==2.2.1 ml-dtypes==0.2",
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
        "pip install jupyter",
        f"pip install --no-deps apache-tvm=={tvm_version} cloudpickle==2.2.1 ml-dtypes==0.2",
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
        "pip install jupyter",
        f"pip install --no-deps apache-tvm=={tvm_version} cloudpickle==2.2.1 ml-dtypes==0.2",
    ]
)

Stage0 += comment("Intel MKL source env")
Stage0 += shell(
    commands=[
        f"conda create -n intel_src python={python_version}",
        "conda activate intel_src",
        "cd numpy",
        "echo -e '[mkl]\\nlibrary_dirs = /opt/intel/compilers_and_libraries_2020/linux/mkl/lib/intel64\\ninclude_dirs = /opt/intel/compilers_and_libraries_2020/linux/mkl/include\\nlibraries = mkl_rt' > site.cfg",
        "pip install 'Cython<3.0.0'",
        "NPY_BLAS_ORDER=mkl NPY_LAPACK_ORDER=mkl python setup.py build -j 4 install",
        "rm -r build",
        "rm site.cfg",
        "pip install jupyter",
        f"pip install --no-deps apache-tvm=={tvm_version} cloudpickle==2.2.1 ml-dtypes==0.2",
    ]
)

Stage0 += comment("Intel MKL Conda env")
Stage0 += shell(
    commands=[
        "conda config --add channels intel",
        f"conda create -n intel_conda intelpython3_core python={python_version}",
        "conda activate intel_conda",
        "pip install jupyter",
        f"pip install --no-deps apache-tvm=={tvm_version} cloudpickle==2.2.1 ml-dtypes==0.2",
    ]
)
