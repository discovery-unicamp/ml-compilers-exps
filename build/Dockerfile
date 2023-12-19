FROM tvm:base

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gfortran \
        git \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# default env

RUN conda create -n default python=3.10.8 && \
    conda activate default && \
    pip install scipy==1.7.3 jupyter && \
    pip install numpy==1.23.5 && \
    pip install --no-deps apache-tvm==0.14.dev16 cloudpickle==2.2.1 ml-dtypes==0.2

RUN git clone --depth 1 --branch v1.23.5 https://github.com/numpy/numpy.git && \
    cd numpy && \
    git submodule update --init && \
    cd .. && \
    git clone --depth 1 --branch v1.7.3 https://github.com/scipy/scipy.git && \
    cd scipy && \
    git submodule update --init

# OpenBLAS env

RUN conda create -n openblas python=3.10.8 && \
    conda activate openblas && \
    cd numpy && \
    echo -e '[openblas]\nlibraries = openblas\nlibrary_dirs = /opt/openblas/lib\ninclude_dirs = /opt/openblas/include\nruntime_library_dirs = /opt/openblas/lib\nextra_link_args = -lm -fopenmp' > site.cfg && \
    pip install 'Cython<3.0.0' && \
    NPY_BLAS_ORDER=openblas NPY_LAPACK_ORDER=openblas python setup.py build -j 4 install && \
    rm -r build && \
    rm site.cfg && \
    pip install jupyter && \
    pip install --no-deps apache-tvm==0.14.dev16 cloudpickle==2.2.1 ml-dtypes==0.2

# AOCL env

RUN conda create -n amd python=3.10.8 && \
    conda activate amd && \
    cd numpy && \
    echo -e '[blis]\nlibraries = blis\nlibrary_dirs = /opt/amd/amd-blis/lib/ILP64\ninclude_dirs = /opt/amd/amd-blis/include/ILP64\nruntime_library_dirs = /opt/amd/amd-blis/lib/ILP64\nextra_link_args = -lm -fopenmp' > site.cfg && \
    echo -e '[flame]\nlibraries = flame\nlibrary_dirs = /opt/amd/amd-libflame/lib/ILP64\ninclude_dirs = /opt/amd/amd-libflame/include/ILP64\nruntime_library_dirs = /opt/amd/amd-libflame/lib/ILP64\nextra_link_args = -lm -fopenmp' >> site.cfg && \
    pip install 'Cython<3.0.0' && \
    NPY_BLAS_ORDER=blis NPY_LAPACK_ORDER=flame python setup.py build -j 4 install && \
    rm -r build && \
    rm site.cfg && \
    pip install jupyter && \
    pip install --no-deps apache-tvm==0.14.dev16 cloudpickle==2.2.1 ml-dtypes==0.2

# AOCL GCC env

RUN conda create -n amd_gcc python=3.10.8 && \
    conda activate amd_gcc && \
    cd numpy && \
    echo -e '[blis]\nlibraries = blis\nlibrary_dirs = /opt/amd_gcc/amd-blis/lib/ILP64\ninclude_dirs = /opt/amd_gcc/amd-blis/include/ILP64\nruntime_library_dirs = /opt/amd_gcc/amd-blis/lib/ILP64\nextra_link_args = -lm -fopenmp -lrt -lpthread' > site.cfg && \
    echo -e '[flame]\nlibraries = flame\nlibrary_dirs = /opt/amd_gcc/amd-libflame/lib/ILP64\ninclude_dirs = /opt/amd_gcc/amd-libflame/include/ILP64\nruntime_library_dirs = /opt/amd_gcc/amd-libflame/lib/ILP64\nextra_link_args = -lm -fopenmp -lgfortran -lquadmath' >> site.cfg && \
    pip install 'Cython<3.0.0' && \
    NPY_BLAS_ORDER=blis NPY_LAPACK_ORDER=flame python setup.py build -j 4 install && \
    rm -r build && \
    rm site.cfg && \
    pip install jupyter && \
    pip install --no-deps apache-tvm==0.14.dev16 cloudpickle==2.2.1 ml-dtypes==0.2

# Intel MKL source env

RUN conda create -n intel_src python=3.10.8 && \
    conda activate intel_src && \
    cd numpy && \
    echo -e '[mkl]\nlibrary_dirs = /opt/intel/compilers_and_libraries_2020/linux/mkl/lib/intel64\ninclude_dirs = /opt/intel/compilers_and_libraries_2020/linux/mkl/include\nlibraries = mkl_rt' > site.cfg && \
    pip install 'Cython<3.0.0' && \
    NPY_BLAS_ORDER=mkl NPY_LAPACK_ORDER=mkl python setup.py build -j 4 install && \
    rm -r build && \
    rm site.cfg && \
    pip install jupyter && \
    pip install --no-deps apache-tvm==0.14.dev16 cloudpickle==2.2.1 ml-dtypes==0.2

# Intel MKL Conda env

RUN conda config --add channels intel && \
    conda create -n intel_conda intelpython3_core python=3.10.8 && \
    conda activate intel_conda && \
    pip install jupyter && \
    pip install --no-deps apache-tvm==0.14.dev16 cloudpickle==2.2.1 ml-dtypes==0.2


