# =============================================================================
# Multi-stage Dockerfile for OpenFOAM 13 + Ginkgo (GPU) + OGL + MixFOAM
# =============================================================================
# Build:   docker build -t mixfoam:latest .
# Multi:   docker build --build-arg CUDA_ARCHS="80;89;120" -t mixfoam:latest .
# Run:     docker run --rm --gpus all -v ./cases:/work -it mixfoam:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder – compile OpenFOAM, Ginkgo, and OGL from source
# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.1.0-devel-ubuntu22.04 AS builder

ARG CUDA_ARCHS="80;89;120"
ENV DEBIAN_FRONTEND=noninteractive

# Build-time dependencies (includes ccache for faster rebuilds and rsync for dev workflow)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        flex \
        libfl-dev \
        bison \
        cmake \
        zlib1g-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libscotch-dev \
        libptscotch-dev \
        libopenmpi-dev \
        openmpi-bin \
        git \
        wget \
        ca-certificates \
        ccache \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# ccache: compiler cache for faster rebuilds.
# With BuildKit cache mounts, ccache persists across docker build invocations.
# Usage: DOCKER_BUILDKIT=1 docker build -t mixfoam:latest .
ENV PATH=/usr/lib/ccache:$PATH \
    CCACHE_DIR=/ccache \
    CCACHE_MAXSIZE=5G

# Allow MPI to run as root inside the container.
# USER must be set so that OpenFOAM's bashrc builds correct FOAM_USER_LIBBIN
# paths (otherwise $USER is empty and paths like /root/OpenFOAM/-13 result).
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    USER=root

# ---- Ginkgo v1.11.0 with CUDA -----------------------------------------------
# Target GPUs: A100 (SM80), Ada 6000 (SM89), RTX 5060 (SM120).
# To add more architectures: "80;89;120;..."
# GIT_SSL_NO_VERIFY works around missing CA bundle in nvidia/cuda base image.
# -j4 limits parallelism to avoid OOM during heavy CUDA template compilation.
RUN GIT_SSL_NO_VERIFY=1 git clone --depth 1 --branch v1.11.0 \
        https://github.com/ginkgo-project/ginkgo.git /tmp/ginkgo-src \
    && cmake -S /tmp/ginkgo-src -B /tmp/ginkgo-build \
        -DCMAKE_INSTALL_PREFIX=/opt/ginkgo \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/usr/local/cuda/lib64/cmake \
        -DGINKGO_BUILD_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DGINKGO_BUILD_REFERENCE=ON \
        -DGINKGO_BUILD_OMP=ON \
        -DGINKGO_BUILD_TESTS=OFF \
        -DGINKGO_BUILD_BENCHMARKS=OFF \
        -DGINKGO_BUILD_EXAMPLES=OFF \
        -DGINKGO_BUILD_DOC=OFF \
    && cmake --build /tmp/ginkgo-build -j4 \
    && cmake --install /tmp/ginkgo-build \
    && rm -rf /tmp/ginkgo-src /tmp/ginkgo-build

# ---- OpenFOAM 13 ----------------------------------------------------------
# Copy the source tree (excluding items listed in .dockerignore)
COPY . /opt/OpenFOAM-13

# Set up the OpenFOAM build environment.
# FOAM_INST_DIR must point to the parent of the OpenFOAM-<version> directory.
# We also override SCOTCH_TYPE to use the system packages installed above.
ENV FOAM_INST_DIR=/opt \
    WM_PROJECT_DIR=/opt/OpenFOAM-13

# Compile OpenFOAM (with ccache for faster rebuilds)
# Allwmake continues past optional components (zoltan, metis, etc.) so we
# cannot use pipefail.  Instead, verify key binaries exist afterward.
# The --mount=type=cache persists ccache across docker build invocations
# (requires DOCKER_BUILDKIT=1).
RUN --mount=type=cache,target=/ccache \
    /bin/bash -c '\
    export FOAM_INST_DIR=/opt && \
    export WM_CONTINUE_ON_ERROR=1 && \
    source /opt/OpenFOAM-13/etc/bashrc SCOTCH_TYPE=system && \
    cd /opt/OpenFOAM-13 && \
    ./Allwmake -j$(( $(nproc) < 32 ? $(nproc) : 32 )) 2>&1 | tee /tmp/openfoam_build.log | tail -80' \
    && test -x /opt/OpenFOAM-13/platforms/linux64GccDPInt32Opt/bin/foamRun \
    && echo "OpenFOAM build verified OK"

# ---- FFT Preconditioner CUDA kernels --------------------------------------
# Compile separately with nvcc before OGL build (gcc cannot compile .cu files).
# The shared library is placed alongside Ginkgo's libs for simple linkage.
RUN nvcc -O3 -shared -Xcompiler -fPIC \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_120,code=sm_120 \
    -o /opt/ginkgo/lib/libfftprecond.so \
    /opt/OpenFOAM-13/modules/OGL/src/OGLSolvers/OGLSolverBase/FFTKernels.cu \
    -lcufft \
    && echo "FFT preconditioner CUDA kernels OK"

# ---- Cylindrical FFT+Thomas CUDA kernels ------------------------------------
# DFT in theta + tridiagonal Thomas solve in r for cylindrical grids.
RUN nvcc -O3 -shared -Xcompiler -fPIC \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_120,code=sm_120 \
    -o /opt/ginkgo/lib/libcylprecond.so \
    /opt/OpenFOAM-13/modules/OGL/src/OGLSolvers/OGLSolverBase/CylFFTKernels.cu \
    -lcufft \
    && echo "Cylindrical FFT preconditioner CUDA kernels OK"

# ---- Halo Exchange CUDA kernels ---------------------------------------------
# GPU gather/scatter kernels for processor-boundary halo exchange.
# No cuFFT dependency — pure CUDA kernels.
RUN nvcc -O3 -shared -Xcompiler -fPIC \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_120,code=sm_120 \
    -o /opt/ginkgo/lib/libhalokernel.so \
    /opt/OpenFOAM-13/modules/OGL/src/OGLOperator/HaloKernels.cu \
    && echo "Halo exchange CUDA kernels OK"

# ---- Geometric Multigrid CUDA kernels ----------------------------------------
# Transfer operators (restriction, trilinear prolongation) and coarse-grid
# CSR matrix construction for geometric MG + FFT preconditioner.
RUN nvcc -O3 -shared -Xcompiler -fPIC \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_120,code=sm_120 \
    -o /opt/ginkgo/lib/libmgkernels.so \
    /opt/OpenFOAM-13/modules/OGL/src/OGLSolvers/OGLSolverBase/MultigridKernels.cu \
    && echo "Geometric MG CUDA kernels OK"

# ---- OGL (OpenFOAM Ginkgo Layer) ------------------------------------------
ENV GINKGO_ROOT=/opt/ginkgo

RUN --mount=type=cache,target=/ccache \
    /bin/bash -o pipefail -c '\
    export FOAM_INST_DIR=/opt && \
    source /opt/OpenFOAM-13/etc/bashrc SCOTCH_TYPE=system && \
    export GINKGO_ROOT=/opt/ginkgo && \
    cd /opt/OpenFOAM-13/modules/OGL && \
    ./Allwmake 2>&1 | tee /tmp/ogl_build.log | tail -20 && echo "OGL OK" || (grep -E "error:" /tmp/ogl_build.log | head -20 && false)'

# ---------------------------------------------------------------------------
# Stage 2: runtime – lean image with only what is needed to run simulations
# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.1.0-devel-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime packages + dev headers needed for incremental OGL compilation
# (libopenmpi-dev provides mpi.h; libscotch-dev for scotch headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libscotch-6.1 \
        libscotch-dev \
        libptscotch-6.1 \
        libopenmpi3 \
        libopenmpi-dev \
        openmpi-bin \
        python3 \
        python3-pip \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python packages for post-processing and the MixFOAM workflow
RUN pip3 install --no-cache-dir \
        numpy \
        matplotlib \
        plotly \
        pyvista \
        vtk \
        jinja2 \
        kaleido

# ---- Copy built artefacts from builder ------------------------------------

# Ginkgo shared libraries + headers (headers needed for dev container workflow)
COPY --from=builder /opt/ginkgo/lib /opt/ginkgo/lib
COPY --from=builder /opt/ginkgo/include /opt/ginkgo/include

# Full OpenFOAM installation (binaries, libraries, etc, tutorials, ...)
COPY --from=builder /opt/OpenFOAM-13 /opt/OpenFOAM-13

# User libraries produced by the build (includes libOGL.so)
COPY --from=builder /root/OpenFOAM /root/OpenFOAM

# ---- Install the mixfoam Python package -----------------------------------
RUN pip3 install --no-cache-dir -e /opt/OpenFOAM-13/mixfoam 2>/dev/null \
    || (echo "mixfoam has no setup.py; adding to PYTHONPATH instead" \
        && echo "/opt/OpenFOAM-13" > /usr/lib/python3/dist-packages/mixfoam.pth)

# ---- Environment -----------------------------------------------------------

ENV WM_PROJECT_DIR=/opt/OpenFOAM-13 \
    FOAM_INST_DIR=/opt \
    GINKGO_ROOT=/opt/ginkgo \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    USER=root

# Library search path: Ginkgo + OpenFOAM platform libs + user libs (OGL)
ENV LD_LIBRARY_PATH=/opt/ginkgo/lib:\
/opt/OpenFOAM-13/platforms/linux64GccDPInt32Opt/lib:\
/opt/OpenFOAM-13/platforms/linux64GccDPInt32Opt/lib/sys-openmpi:\
/root/OpenFOAM/root-13/platforms/linux64GccDPInt32Opt/lib:\
${LD_LIBRARY_PATH}

# Put OpenFOAM binaries and scripts on PATH
ENV PATH=/opt/OpenFOAM-13/platforms/linux64GccDPInt32Opt/bin:\
/opt/OpenFOAM-13/bin:\
/opt/OpenFOAM-13/wmake:\
${PATH}

# Source OpenFOAM bashrc for interactive (login) shells
RUN echo 'export FOAM_INST_DIR=/opt && . /opt/OpenFOAM-13/etc/bashrc' >> /etc/bash.bashrc

WORKDIR /work

# Entrypoint: source OpenFOAM environment then execute whatever is requested
ENTRYPOINT ["/bin/bash", "-c", "export FOAM_INST_DIR=/opt && source /opt/OpenFOAM-13/etc/bashrc && exec \"$@\"", "--"]
CMD ["bash"]
