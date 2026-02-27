# OGL — OpenFOAM Ginkgo Layer

GPU-accelerated linear solvers for OpenFOAM via the
[Ginkgo](https://ginkgo-project.github.io/) numerical linear algebra library.

## Solvers

| Solver | System | Preconditioners |
|--------|--------|-----------------|
| `OGLPCG` | Symmetric (pressure) | AMG, Block-Jacobi, ILU-ISAI, Jacobi, none |
| `OGLBiCGStab` | Asymmetric (momentum) | Block-Jacobi, ILU-ISAI, Jacobi, none |

The AMG preconditioner uses Ginkgo's parallel graph-matching (PGM)
coarsening with configurable smoothers and hierarchy caching.

## Dependencies

- OpenFOAM 13
- [Ginkgo](https://github.com/ginkgo-project/ginkgo) >= 1.11.0 (CUDA backend)
- CUDA toolkit (for cuFFT + cuSPARSE)

Set `GINKGO_ROOT` to the Ginkgo installation prefix before building.

## Build

```bash
export GINKGO_ROOT=/path/to/ginkgo
cd modules/OGL
./Allwmake
```

## Configuration

Add to `system/controlDict`:
```
libs ("libOGL.so");
```

Pressure solver in `system/fvSolution`:
```
p
{
    solver          OGLPCG;
    tolerance       1e-6;
    relTol          0.01;

    OGLCoeffs
    {
        preconditioner  multigrid;
        mgCacheInterval 10;          // Rebuild hierarchy every N solves
        cacheStructure  true;
        cacheValues     true;
    }
}
```

Momentum solver in `system/fvSolution`:
```
"(U|k|epsilon)"
{
    solver          OGLBiCGStab;
    tolerance       1e-5;
    relTol          0.1;

    OGLCoeffs
    {
        preconditioner  ilu_isai;    // or blockJacobi
        isaiPower       1;
    }
}
```

See `etc/fvSolution.example` for all options.

## Preconditioner selection guide

| Mesh size | Pressure | Momentum |
|-----------|----------|----------|
| < 100K cells | CPU GAMG (GPU overhead dominates) | CPU smoothSolver |
| 100K–4.5M | `OGLPCG` + AMG | `OGLBiCGStab` + Block-Jacobi |
| > 4.5M | `OGLPCG` + AMG | `OGLBiCGStab` + ILU-ISAI |

For variable-coefficient problems (MRF, turbulence), set `mgCacheInterval 10`
to maintain AMG hierarchy quality. Constant-coefficient problems can use
larger intervals.

## Hardware compatibility

Ginkgo provides backend abstraction across:
- NVIDIA GPUs (CUDA)
- AMD GPUs (HIP)
- Intel GPUs (DPC++)

Validated on RTX 5060 (8 GB GDDR7, SM 12.0) with Ginkgo 1.11.0.

## License

GPL-3.0, consistent with OpenFOAM.
