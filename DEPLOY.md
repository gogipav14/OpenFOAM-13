# MixFOAM Docker Deployment

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- BuildKit enabled (`DOCKER_BUILDKIT=1`)
- ~200 GB free disk space for the build

## Build

```bash
git clone <repo-url> OpenFOAM-13
cd OpenFOAM-13

# A100 only (SM80) — fastest build
DOCKER_BUILDKIT=1 docker build --build-arg CUDA_ARCHS="80" -t mixfoam:latest .

# Multi-GPU (A100 + Ada 6000 + RTX 5060)
DOCKER_BUILDKIT=1 docker build --build-arg CUDA_ARCHS="80;89;120" -t mixfoam:latest .
```

Build takes ~45–60 min. Full output is saved inside the build layer at
`/tmp/openfoam_build.log` — if the build fails, inspect it with:

```bash
docker run --rm mixfoam:latest cat /tmp/openfoam_build.log | grep -E "error:|Error " | head -30
```

## Verify

```bash
docker run --rm --gpus all mixfoam:latest foamRun -help
docker run --rm --gpus all mixfoam:latest nvidia-smi
docker run --rm --gpus all mixfoam:latest bash -c \
    'ls $FOAM_USER_LIBBIN/libOGL.so && echo "OGL OK"'
```

## Run a Case

```bash
# Copy a case outside the repo (cases are mounted, not baked into the image)
mkdir -p cases
cp -r mixing_cases/sartorius_50L_benchmark cases/sartorius

# Run
docker run --rm --gpus all \
    -v $(pwd)/cases/sartorius:/work \
    mixfoam:latest ./Allrun
```

## GPU Solver Configuration

Edit `system/fvSolution` in your case directory. Example for GPU pressure + momentum:

```
libs ("libOGL.so");

solvers
{
    p
    {
        solver          OGLPCG;
        preconditioner  BLOCK_JACOBI;
        tolerance       1e-6;
        relTol          0.01;
    }

    U
    {
        solver          OGLBiCGStab;
        preconditioner  ILU;
        tolerance       1e-5;
        relTol          0.1;
    }
}
```

See `modules/OGL/etc/fvSolution.example` for all solver options and preconditioner types.

## Dev Container (iterative OGL development)

For rapid edit-compile-test cycles without rebuilding the full image:

```bash
./dev.sh start        # Create persistent container with build caches
./dev.sh compile      # Incremental OGL compile (~30-60s)
./dev.sh shell        # Interactive shell
./dev.sh stop         # Tear down
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `cannot find -lOpenFOAM` during build | Core libs failed to compile. Check `/tmp/openfoam_build.log` inside the failed layer. Likely OOM — reduce `-j` cap in Dockerfile. |
| `nvidia-smi` not found in container | Install NVIDIA Container Toolkit on the host. |
| `libOGL.so` missing | Check that `GINKGO_ROOT=/opt/ginkgo` is set and Ginkgo built successfully. |
| Slow Docker context transfer | Ensure `.dockerignore` excludes `benchmarks/`, `benchmark_results_full/`, etc. |

## Architecture Notes

The Dockerfile is a two-stage build:

1. **builder** — compiles Ginkgo, OpenFOAM, CUDA kernels (FFT + halo), and OGL
2. **runtime** — copies built artifacts into a lean image with Python post-processing tools

The `SCOTCH_TYPE=system` override is passed as an argument to `etc/bashrc`
(not as an env var before sourcing) because `etc/bashrc:106` unconditionally
sets `SCOTCH_TYPE=ThirdParty`. OpenFOAM's `_foamParams` mechanism processes
bashrc arguments after the hardcoded exports, making this the correct override
method.
