Stage0 += baseimage(image="ubuntu:22.04")

Stage0 += raw(
    docker='SHELL ["/bin/bash", "--login", "-c"]', singularity="# no equivalent"
)

Stage0 += apt_get(ospackages=["build-essential"])
Stage0 += mkl(eula=True)
Stage0 += openblas(
    prefix="/opt/openblas", make_opts=["TARGET=SKYLAKEX", "USE_OPENMP=1"]
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
