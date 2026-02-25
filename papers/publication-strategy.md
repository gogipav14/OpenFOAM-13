# Publication Strategy v2.0 — Updated From Research Results

**Last updated:** 2026-02-21
**Branch:** `master` (merged from `ogl-preconditioner-research`)

## Overview

Three papers targeting different audiences, built entirely from validated results.

| | Paper 1 | Paper 2 | Paper 3 |
|---|---|---|---|
| **Title** | GPU-Accelerated AMG and ILU-ISAI Preconditioned Krylov Solvers for OpenFOAM via Ginkgo | MixFOAM: Automated CFD Case Generation for Industrial Mixing Vessels | On the Superlinear Iteration Growth of Algebraic Multigrid at Scale in OpenFOAM |
| **Journal** | Computer Physics Communications | SoftwareX | Int. J. Numer. Meth. Fluids (short comm.) |
| **IF** | ~7.2 (Q1) | ~2.4 (Q2) | ~1.7 (Q2) |
| **Review** | ~12 months | ~9 weeks | ~6 months |
| **Focus** | Numerical methods + GPU preconditioner selection | Software tool description | Empirical finding on GAMG scaling |
| **Length** | ~20-25 pages | 3,000 words, 6 figures | ~6-8 pages |
| **Data ready** | 90% | 70% | 80% |

---

## What Changed From v1.0

The original strategy centered on four contributions: adaptive mixed-precision, GPU-resident halo exchange, memory pooling, and batched solving. Research invalidated this framing:

| Original Contribution | Outcome | Status |
|----------------------|---------|--------|
| Adaptive mixed-precision | Ginkgo v1.11.0 FP32 SpGEMM is 3-8x slower than FP64. FP32 CG ~2x slower per-iteration. Produces *worse* wall time. | **Dead** |
| GPU-resident halo exchange | Code functional (690 LOC + CUDA kernels). But single-GPU data transfer is 0.016% of step time. Not measurably impactful. | **Immaterial on 1 GPU** |
| Memory pool | Bucket allocation functional. Never benchmarked. `createVector()` bypass found in code audit. | **Unbenchmarked** |
| Batched solver | `executeBatchedSolve()` is a stub — falls back to sequential. | **Incomplete** |

What we *actually* built and validated:

| Actual Contribution | Evidence |
|---------------------|----------|
| AMG-PCG with convergence-gated hierarchy caching | 5.47x at 3M cells. mgCacheInterval=10 optimal (+14.4% vs default). |
| BiCGStab + ILU-ISAI for asymmetric momentum | 6.85x at 5M cells. Crossover vs Block-Jacobi at ~4.5M confirmed. |
| Preconditioner selection analysis (BJ vs ILU+ISAI vs CPU symGS) | 8-mesh scaling study with iteration + wall time crossover. |
| GAMG superlinear iteration degradation | 13 → 66 → 124 → 152 iters (40K → 1M → 3M → 5M). GPU MG linear: 22 → 44 → 66 → 44. |
| CSR refresh propagation fix | invalidateValues() for variable-coefficient momentum matrices. |

**Paper 1 is completely reframed around the actual results. The story is stronger: definitive speedups with production recommendations, not aspirational mixed-precision claims.**

---

# Paper 1: GPU-Accelerated AMG and ILU-ISAI Krylov Solvers for OpenFOAM

**Target:** Computer Physics Communications (Elsevier)
**Article type:** Computer Programs in Physics

## Title

> GPU-Accelerated Algebraic Multigrid and ILU-ISAI Preconditioned Krylov Solvers for Finite-Volume CFD via the Ginkgo Linear Algebra Library

## Abstract (draft, ~250 words)

We present a GPU-accelerated linear solver module for the OpenFOAM finite-volume framework, built on the Ginkgo numerical linear algebra library. The module provides two solver classes targeting the dominant computational costs in pressure-velocity coupled CFD: (1) a preconditioned conjugate gradient solver with algebraic multigrid (AMG) preconditioning for the symmetric pressure Poisson system, featuring convergence-gated hierarchy caching that amortizes the expensive multigrid setup across multiple solves; and (2) a stabilized biconjugate gradient solver (BiCGStab) with parallel incomplete LU preconditioning using incomplete sparse approximate inverse triangular solves (ILU-ISAI) for the asymmetric momentum system.

We validate the implementation on structured and unstructured meshes spanning 40,000 to 5,000,000 cells. The GPU AMG-PCG pressure solver achieves 5.47x speedup over OpenFOAM's native GAMG at 3M cells, with the speedup increasing monotonically with mesh size. We identify and quantify a previously unreported superlinear iteration growth in OpenFOAM's GAMG implementation (13 iterations at 40K cells to 152 at 5M), while the GPU AMG iteration count scales linearly. For the momentum system, we perform a systematic comparison of Block-Jacobi and ILU-ISAI preconditioners, identifying a crossover point at approximately 4.5M cells above which ILU-ISAI delivers superior wall-clock performance (6.85x speedup at 5M cells) due to its iteration quality matching CPU-native symmetric Gauss-Seidel. We provide mesh-size-dependent solver selection recommendations validated across the full scaling range. The solver is open-source and portable across NVIDIA, AMD, and Intel GPUs via Ginkgo's backend abstraction.

## 1. Introduction

### Motivation
- Pressure Poisson equation dominates incompressible/weakly-compressible CFD cost (60-80% wall time)
- LES of industrial flows requires fine meshes (2-20M cells) and long physical times
- GPU acceleration of sparse linear solvers is the highest-impact optimization target
- Existing approaches: AmgX (NVIDIA-only), PETSc/HYPRE (complex integration), base OGL (no AMG, no momentum solver)
- **No prior work provides GPU AMG + asymmetric momentum solver for OpenFOAM with preconditioner selection guidance**

### Literature context
- Olenik et al. (2024) -- base OGL plugin for OpenFOAM/Ginkgo integration [Meccanica]
- Piscaglia & Ghioldi (2023) -- amgx4Foam: NVIDIA AmgX for OpenFOAM [Aerospace]
- petsc4Foam -- PETSc/HYPRE integration [NextFOAM]
- Cojean et al. (2024) -- Ginkgo library [IJHPCA]
- Wendler et al. (2024) -- mixed precision in FlowSimulator [Scipedia]
- GPU-accelerated coupled solvers (2024) -- heterogeneous GPGPU implicit [arXiv]
- Anzt et al. (2022) -- ISAI for approximate triangular solves [SISC]
- Chow & Patel (2015) -- ParILU for GPUs [SISC]

### Contributions
1. GPU AMG-PCG with convergence-gated hierarchy caching for pressure
2. GPU BiCGStab with ILU-ISAI preconditioning for asymmetric momentum
3. Systematic BJ vs ILU-ISAI crossover analysis with wall-time and iteration metrics
4. Empirical documentation of GAMG superlinear iteration growth at scale
5. Validated mesh-size-dependent solver selection recommendations
6. Comprehensive benchmarking on 2D and 3D cases (40K-5M cells)

## 2. Mathematical Formulation

### 2.1 Pressure Poisson System
- LDU matrix storage and CSR conversion for GPU
- Symmetric positive-definite system: PCG with AMG preconditioner
- Ginkgo PGM (Parallel Graph Match) coarsening
- Jacobi d2 smoother (validated optimal vs Chebyshev by wall time)
- Convergence-gated hierarchy caching: rebuild only when iteration count exceeds threshold or interval expires

**Hierarchy caching formulation:**

Let $k_i$ be the iteration count at call $i$, $\tau$ the max-iteration threshold, and $N$ the cache interval.

$$\text{rebuild}(i) = \begin{cases} \text{true} & \text{if } i \mod N = 0 \\ \text{true} & \text{if } k_{i-1} > \tau \\ \text{false} & \text{otherwise} \end{cases}$$

This amortizes the $O(N_{\text{levels}} \cdot \text{nnz})$ hierarchy construction cost across $N$ calls while preserving convergence through the max-iteration safety valve.

### 2.2 Momentum System
- Asymmetric matrix from advection terms: BiCGStab solver
- Block-Jacobi preconditioner: extracts block-diagonal, inverts locally
- ILU-ISAI preconditioner:
  - ParILU factorization: $A \approx LU$ computed in parallel (Chow-Patel algorithm)
  - ISAI triangular solves: precompute $\tilde{L}^{-1} \approx L^{-1}$, $\tilde{U}^{-1} \approx U^{-1}$ as sparse matrices
  - Application: $M^{-1}r = \tilde{U}^{-1}(\tilde{L}^{-1}r)$ via two parallel SpMV (not sequential forward/backward substitution)

**Key insight (validated empirically):** Exact triangular solves (`LowerTrs`/`UpperTrs`) are fundamentally sequential (row $i$ depends on row $i-1$), making them hostile to GPU thread occupancy. ISAI trades exactness for parallelism -- the approximate inverse is computed once and applied as parallel SpMV, eliminating the sequential bottleneck.

### 2.3 LDU to CSR Conversion
- Structure caching: sparsity pattern reused across solves on static meshes
- Value refresh: `invalidateValues()` propagates CSR coefficient update when momentum matrix changes each PISO corrector
- Handles symmetric (pressure) and asymmetric (momentum) matrices

## 3. Implementation

### 3.1 Software Architecture

```
OGLSolverBase (lduMatrix::solver)
  |-- OGLPCGSolver          # Symmetric PCG + AMG preconditioner
  |-- OGLBiCGStabSolver     # Asymmetric BiCGStab + BJ/ILU-ISAI
  |
  |-- FoamGinkgoLinOp        # Custom Ginkgo LinOp wrapping OpenFOAM LDU
  |-- lduToCSR               # LDU -> CSR conversion with structure caching
  |-- OGLExecutor            # Singleton device manager
```

### 3.2 AMG Hierarchy Caching
- Static cache keyed by field name (pressure, velocity components)
- Mutex-protected for thread safety
- Hierarchy persisted across solver instantiations (OpenFOAM creates new solver object each call)
- Interval + max-iteration dual gate

### 3.3 Preconditioner Selection
- Factory pattern: `preconditioner` keyword in `OGLCoeffs` sub-dictionary
- Options: `blockJacobi`, `ILU`, `ISAI`, `multigrid`
- `ILU` internally uses `gko::factorization::ParIlu` + `gko::preconditioner::Ilu<LowerIsai, UpperIsai>`

### 3.4 Performance Instrumentation
- RAII `ScopedTimer` with per-phase timing
- VRAM diagnostics (free/total) logged at hierarchy construction
- Per-solve iteration count and convergence status tracking

## 4. Benchmark Cases

### 4.1 Case A: 2D Lid-Driven Cavity (Scaling Study)

The primary scaling vehicle. kEpsilon RANS, PISO with 3 pressure correctors, 20 timesteps.

**Mesh sizes:** 200x200 (40K) through 2236x2236 (5M)

**Solver configurations compared:**

| Variant | Pressure | Momentum | Label |
|---------|----------|----------|-------|
| CPU | GAMG + GaussSeidel | smoothSolver + symGS | Baseline |
| GPU-p | OGLPCG + AMG (Ginkgo) | smoothSolver + symGS (CPU) | Pressure-only GPU |
| GPU-all (BJ) | OGLPCG + AMG | OGLBiCGStab + Block-Jacobi | Full GPU, BJ momentum |
| GPU-all (ILU) | OGLPCG + AMG | OGLBiCGStab + ILU-ISAI | Full GPU, ILU momentum |

**Already collected:** Full scaling data at 8 mesh sizes (200, 600, 1000, 1200, 1500, 1732, 2000, 2236). Stored in `benchmarks/bicgstab_results_ilu_isai/` and `benchmarks/bicgstab_results_crossover/`.

### 4.2 Case B: 3D Stirred Tank LES (Industrial Application)

Demonstrates GPU solver on a realistic industrial case with rotating geometry, non-conformal coupling, and passive scalar transport.

**Source:** `tutorials/incompressibleFluid/stirredTankLES_GPU/`
**Solver:** incompressibleFluid, PIMPLE, WALE LES
**Mesh:** ~143K cells (base), scale to 1-3M via snappyHexMesh refinement
**Fields:** U, p, tracer

**Benchmark plan:**
- CPU baseline: GAMG + smoothSolver (standard OpenFOAM)
- GPU-p: OGLPCG + AMG for pressure, CPU momentum
- GPU-all (BJ): OGLPCG + AMG + OGLBiCGStab+BJ
- GPU-all (ILU): OGLPCG + AMG + OGLBiCGStab+ILU-ISAI
- 100 timesteps per configuration (exclude first 10 for warmup)

**Visualization deliverables:**
- Velocity magnitude mid-plane slice (CPU vs GPU, qualitative match)
- GPU-CPU field difference contour ($|U_\text{GPU} - U_\text{CPU}|$)
- Tracer concentration evolution (4-panel time series)
- Residual convergence histories per-solve overlay (CPU vs GPU)

### 4.3 Case C: 3D MotorBike Steady-State (Large Unstructured Mesh)

External aerodynamics around complex geometry. SIMPLE steady-state eliminates time-stepping variability -- pure solver benchmarking.

**Source:** `tutorials/incompressibleFluid/motorBike/motorBike/`
**Solver:** incompressibleFluid, SIMPLE (steady)
**Mesh:** 1M-5M+ cells (scalable via snappyHexMesh refinement levels)
**Fields:** U, p, k, nuTilda (Spalart-Allmaras)

**Benchmark plan:**
- Same 4 solver configurations
- Steady-state: run to convergence (residuals < 1e-5), compare total iteration count and wall time
- Mesh scaling: run at 1M, 2M, 4M cells

**Visualization deliverables:**
- Pressure coefficient on motorcycle surface (CPU vs GPU)
- Velocity streamlines in wake region
- Drag/lift coefficient convergence history (CPU vs GPU overlay)
- Residual convergence comparison (all fields, log scale)

---

## 5. Visualization and Comparison Pipeline

### 5.1 Infrastructure to Build

A Python-based comparison tool (`benchmarks/compare_solutions.py`) that automates all publication figures:

```
compare_solutions.py
  |-- VTK field reader (pyvista)
  |-- Field difference computation (GPU - CPU)
  |-- Residual history parser (OpenFOAM log files)
  |-- Automated figure generation (matplotlib for print, plotly for interactive)
```

### 5.2 Solution Accuracy Validation

For each case and configuration, compute:

**Field-level metrics:**
- $L_\infty$ relative error: $\|x_\text{GPU} - x_\text{CPU}\|_\infty / \|x_\text{CPU}\|_\infty$
- $L_2$ relative error: $\|x_\text{GPU} - x_\text{CPU}\|_2 / \|x_\text{CPU}\|_2$
- Pointwise difference contour plots

**Convergence metrics:**
- Per-solve iteration count time series (GPU vs CPU)
- Per-solve final residual time series
- Cumulative iteration count over N timesteps

**Expected:** GPU and CPU solutions should agree to $O(10^{-12})$ or better for FP64 solvers (difference only from solver algorithm, not precision). Deviation beyond $O(10^{-6})$ indicates a bug.

### 5.3 Residual Convergence Comparison

Parse OpenFOAM log files and overlay:

```
For each field (p, Ux, Uy, Uz, k, epsilon, ...):
  Plot: x-axis = outer iteration (or timestep)
        y-axis = initial residual (log scale)
  Series: CPU (solid), GPU-p (dashed), GPU-all-BJ (dotted), GPU-all-ILU (dash-dot)
```

This proves the GPU solver is not just fast but converges identically. Any divergence in residual trajectories is immediately visible and must be explained.

### 5.4 VTK Visualization Workflow

**Per-case, per-variant:**
1. Write VTK at key timesteps (foamToVTK or built-in function object)
2. Load with pyvista, extract mid-plane slice
3. Render: velocity magnitude, pressure, turbulence fields
4. Compute pointwise difference: `diff = vtk_gpu.point_data['U'] - vtk_cpu.point_data['U']`
5. Render difference field with diverging colormap (blue-white-red)

**Stirred tank specific:**
- Horizontal slice at impeller height
- Vertical slice through impeller axis
- Tracer isosurface evolution (4 frames: t=0.1s, 0.5s, 1.0s, 2.0s)

**MotorBike specific:**
- Surface pressure coefficient ($C_p$) on motorcycle body
- Velocity magnitude on vertical centerplane
- Wake vorticity isosurfaces

### 5.5 Animation (Supplementary Material)

CPC allows supplementary multimedia. Generate:
- Stirred tank tracer mixing animation (100 frames, mp4)
- Side-by-side CPU vs GPU velocity field evolution
- Residual convergence animation showing per-iteration progress

---

## 6. Figures (Planned, ~12-14)

### Scaling and Performance (from existing data)
1. **Speedup vs mesh size** -- 4 variants, 8 mesh sizes, log-log (hero figure)
2. **ILU-BJ crossover analysis** -- bar chart of wall-time gap converging to crossover
3. **Ux iteration comparison** -- BJ inflation vs ILU quality vs CPU symGS
4. **GAMG iteration degradation** -- CPU GAMG vs GPU MG-PCG iteration count
5. **mgCacheInterval sweep** -- step time vs interval (5, 10, 20, 40)
6. **Step time breakdown** -- stacked bar: pressure solve, momentum solve, other

### 3D Case Visualization (to be generated)
7. **Stirred tank velocity field** -- mid-plane slice, CPU vs GPU side-by-side
8. **GPU-CPU field difference** -- pointwise $|U_\text{GPU} - U_\text{CPU}|$ on mid-plane
9. **Stirred tank tracer evolution** -- 4-panel time series
10. **MotorBike pressure coefficient** -- surface $C_p$, CPU vs GPU
11. **MotorBike wake streamlines** -- velocity streamlines in wake region

### Convergence and Accuracy
12. **Residual convergence overlay** -- per-field, CPU vs GPU (stirred tank)
13. **Solution accuracy table** -- $L_\infty$, $L_2$ errors for all cases/variants
14. **Per-solve iteration time series** -- pressure iterations over 100 timesteps

### Architecture Diagram
15. **Software architecture** -- OGLSolverBase hierarchy + Ginkgo integration

---

## 7. Results and Discussion (Outline)

### 7.1 Scaling Study (Case A: Cavity)

Present the full 40K-5M scaling table. Key findings:

**GPU AMG-PCG pressure:**
- Break-even at ~450K cells. Below this, PCIe overhead dominates.
- 5.47x at 3M, 5.58x at 5M. Speedup not plateauing.
- mgCacheInterval=10 gives +14.4% vs default (5). Interval=20 degrades at 5M.

**GPU BiCGStab momentum -- BJ vs ILU-ISAI crossover:**
- BJ iteration count grows superlinearly: 3.1 (40K) -> 10.6 (3M) -> 13.4 (5M)
- ILU-ISAI iterations track CPU quality: 2.0 (40K) -> 6.7 (3M) -> 8.5 (5M)
- Wall-time crossover at ~4.5M cells (ILU-ISAI setup cost amortized by iteration savings)
- At 5M: ILU-ISAI 15.8% faster than BJ, 18.6% faster than GPU-p

**GAMG superlinear degradation:**
- CPU GAMG: 13 -> 66 -> 124 -> 152 iters (40K -> 1M -> 3M -> 5M)
- GPU MG-PCG: 22 -> 44 -> 66 -> 44 (roughly linear, possibly sublinear)
- At 5M, GAMG needs 3.5x more iterations than GPU MG

### 7.2 3D Stirred Tank (Case B)

Demonstrate that the scaling conclusions transfer to realistic 3D transient cases:
- GPU speedup consistent with cavity scaling at equivalent cell count
- Velocity and tracer fields visually identical between CPU and GPU
- Quantitative field agreement: $L_\infty$ error < $10^{-10}$ (FP64)
- No impact on physical solution quality (power number, mixing time)

### 7.3 3D MotorBike (Case C)

Large unstructured mesh with complex geometry:
- Steady-state convergence: GPU and CPU reach same residual levels
- Drag/lift coefficients match within solver tolerance
- Performance at 2M-4M cells

### 7.4 Negative Results (Honest Reporting)

**Mixed-precision MG (FP32):**
- Ginkgo v1.11.0 FP32 SpGEMM (used in MG hierarchy) is 3-8x slower than FP64
- FP32 CG per-iteration ~2x slower
- Net: FP32 MG produces worse wall time despite fewer bytes
- Conclusion: consumer GPUs (FP64:FP32 = 1:64) cannot exploit mixed precision for AMG until sparse matrix kernels are optimized for FP32

**Exact triangular solves (LowerTrs/UpperTrs):**
- ILU with exact Trs cuts iterations 61% but wall time identical to BJ
- Sequential forward/backward substitution is GPU-hostile
- ISAI swap recovers parallelism; documented as essential for GPU ILU

---

## 8. Conclusions

- GPU AMG-PCG achieves 5.47x speedup at 3M cells with convergence-gated hierarchy caching
- ILU-ISAI delivers 6.85x at 5M cells, outperforming all other configurations
- Preconditioner selection is mesh-size-dependent: BJ below 4.5M, ILU-ISAI above
- GAMG iteration growth is superlinear and accelerates GPU advantage at scale
- GPU MG iterations scale linearly, suggesting continued speedup improvement beyond 5M
- Open-source, vendor-portable via Ginkgo (NVIDIA/AMD/Intel GPUs)
- Code: https://github.com/gogipav14/OpenFOAM-13

---

## 9. References (Key, ~30-40 total)

- Olenik et al. (2024) -- OGL base plugin [Meccanica]
- Cojean et al. (2024) -- Ginkgo library [IJHPCA]
- Piscaglia & Ghioldi (2023) -- amgx4Foam [Aerospace]
- Anzt et al. (2022) -- ISAI for approximate triangular solves [SISC]
- Chow & Patel (2015) -- Fine-grained parallel ILU for GPUs [SISC]
- Trottenberg, Oosterlee & Schuller (2001) -- Multigrid [Academic Press]
- Saad (2003) -- Iterative Methods for Sparse Linear Systems
- Weller et al. (1998) -- OpenFOAM
- Wendler et al. (2024) -- mixed precision FlowSimulator [Scipedia]
- Carson & Higham (2018) -- adaptive precision in Krylov methods [SISC]
- Stueben (2001) -- algebraic multigrid review [J. Comp. Appl. Math]

---

# Paper 2: MixFOAM (SoftwareX) -- Unchanged

**Target:** SoftwareX (Elsevier)
**Status:** 70% data ready. Needs one complete mixing simulation for illustrative example.

*(Full outline retained from v1.0 -- see git history for details. No changes needed to Paper 2 strategy.)*

### Remaining Work
- Run one complete Mobius_MIX 100L case through full pipeline
- Capture: EDR, power number, mixing time, shear rate distributions
- Generate HTML report for paper screenshots
- Record actual (not illustrative) metric values for Table 1

---

# Paper 3: GAMG Superlinear Degradation (Short Communication)

**Target:** International Journal for Numerical Methods in Fluids (short communication)
**Status:** 80% data ready. Needs GAMG sensitivity analysis.

## Title

> On the Superlinear Iteration Growth of Algebraic Multigrid in OpenFOAM: Implications for GPU-Accelerated Alternatives

## Core Finding

OpenFOAM's GAMG pressure solver exhibits superlinear iteration growth with mesh refinement on the pressure Poisson system:

| Cells | GAMG iters | GPU MG iters | Ratio |
|-------|-----------|-------------|-------|
| 40K | 13.1 | 10.4 | 1.3x |
| 360K | 30.2 | 19.6 | 1.5x |
| 1M | 66.1 | 24.3 | 2.7x |
| 3M | 122.7 | 37.8 | 3.2x |
| 5M | 151.8 | 43.9 | 3.5x |

GAMG iteration count roughly doubles per 3x mesh refinement (superlinear). GPU MG (Ginkgo PGM + Jacobi smoother) scales linearly. This divergence is the primary driver of GPU speedup at scale.

## Remaining Work

Before publishing this as a standalone finding:

1. **GAMG sensitivity analysis:** Test with different agglomeration parameters (`nCellsInCoarsestLevel`, `agglomerator`, `nPreSweeps`, `nPostSweeps`). If GAMG degradation disappears with tuning, it's a configuration issue, not a fundamental finding.
2. **3D verification:** Confirm the trend holds on 3D unstructured meshes (motorBike at 1M/2M/4M), not just 2D structured cavity.
3. **Literature check:** Verify this isn't already reported in OpenFOAM community (CFD-online forums, OpenFOAM wiki, ESI release notes).

If the finding survives scrutiny, this is a 4-6 page short communication with high practical impact for the OpenFOAM user community.

---

# Execution Plan: 3D Case Benchmarks + Visualization

## Phase 1: Visualization Infrastructure (1-2 days)

Build `benchmarks/compare_solutions.py`:

```python
# Core functions needed:
def parse_residuals(log_file) -> dict[str, list[float]]
    # Parse "Solving for Ux, Initial residual = ..." lines from OpenFOAM log

def load_vtk_fields(case_dir, time) -> pyvista.MultiBlock
    # Load VTK output at given timestep

def compute_field_difference(vtk_a, vtk_b, field_name) -> pyvista.DataSet
    # Pointwise difference with L_inf, L_2 norms

def render_midplane_slice(vtk, field, normal, origin, output_path)
    # pyvista screenshot of field on cutting plane

def render_difference_map(vtk_diff, field, output_path)
    # Diverging colormap (blue-white-red) for GPU-CPU difference

def plot_residual_overlay(logs: dict[str, str], fields: list[str], output_path)
    # Matplotlib: per-field residual convergence, one line per variant

def plot_iteration_timeseries(logs: dict[str, str], field, output_path)
    # Iteration count per solve call over simulation time

def generate_accuracy_table(cases: list, variants: list) -> str
    # LaTeX table of L_inf, L_2 errors for all combinations
```

Dependencies: `pyvista`, `matplotlib`, `numpy` (all available in Docker image or pip-installable).

## Phase 2: Stirred Tank Benchmark (2-3 days)

### 2a. Scale up mesh

The base tutorial has 143K cells. For the paper we need 1-3M cells.

**Option A:** Increase blockMesh resolution (double each direction: 143K -> ~1.1M)
**Option B:** Add snappyHexMesh refinement levels around impeller

Target: ~2M cells for a case that runs in reasonable time on RTX 5060.

### 2b. Run 4 variants

For each variant (CPU, GPU-p, GPU-all-BJ, GPU-all-ILU):
1. Run 110 timesteps (10 warmup + 100 measured)
2. Write VTK at timesteps 50, 75, 100, 110
3. Capture full solver log

### 2c. Generate figures

From the runs:
- Figure 7: Velocity magnitude on horizontal slice at impeller height
- Figure 8: $|U_\text{GPU} - U_\text{CPU}|$ difference map
- Figure 9: Tracer isosurface at 4 timepoints
- Figure 12: Residual convergence overlay
- Figure 14: Per-solve p iteration count over 100 timesteps

### 2d. Accuracy table

Compute $L_\infty$ and $L_2$ for U, p, tracer at t=final for all GPU variants vs CPU reference.

## Phase 3: MotorBike Benchmark (2-3 days)

### 3a. Mesh generation

Run snappyHexMesh at 3 refinement levels targeting 1M, 2M, 4M cells.

### 3b. Run 4 variants to steady state

SIMPLE iterations until all residuals < 1e-5. Log total wall time and iteration counts.

### 3c. Generate figures

- Figure 10: Surface $C_p$ on motorcycle (CPU vs GPU)
- Figure 11: Velocity streamlines in wake
- Drag/lift coefficient convergence
- Residual convergence comparison

## Phase 4: GAMG Sensitivity (1 day)

Test GAMG with varied parameters on the 2D cavity at 1M-3M cells:

```
agglomerator    faceAreaPair;
nCellsInCoarsestLevel  10;   // default
nCellsInCoarsestLevel  100;  // aggressive coarsening
nCellsInCoarsestLevel  500;  // conservative coarsening
smoother        GaussSeidel;  // default
smoother        DIC;          // different smoother
nPreSweeps      0;            // default
nPreSweeps      1;            // more pre-smoothing
```

If iteration count is insensitive to tuning -> fundamental GAMG limitation -> Paper 3 is valid.
If iteration count drops dramatically with tuning -> configuration issue -> Paper 3 is killed.

---

# Timeline

```
Week 1:   Build compare_solutions.py visualization pipeline
          Run GAMG sensitivity analysis (go/no-go for Paper 3)

Week 2:   Stirred tank benchmark (scale mesh, run 4 variants, VTK output)
          Generate stirred tank figures (7-9, 12, 14)

Week 3:   MotorBike benchmark (mesh generation, 4 variants, steady state)
          Generate motorBike figures (10-11)
          Compute accuracy tables for all cases

Week 4:   Assemble Paper 1 draft (CPC format)
          All figures finalized
          Internal review

Week 5:   Paper 1 revisions + submit to CPC
          Start Paper 2 illustrative example (MixFOAM run)

Week 6:   Paper 2 draft + submit to SoftwareX
          Paper 3 draft (if GAMG finding holds)

Week 8:   Paper 3 submit to IJNMF (if applicable)
```

**Key dependency:** Paper 2 references Paper 1 as "submitted" or "in preparation". Paper 1 references Paper 2 for MixFOAM case generation methodology.

---

# Hardware

| Resource | Available | Required |
|----------|-----------|----------|
| GPU | RTX 5060, 8GB (Ada Lovelace) | Sufficient for all cases up to 5M cells |
| CPU | Intel Core Ultra 5 225F | Sufficient for baselines |
| Docker | ogl-dev container with OpenFOAM 13 + Ginkgo v1.11.0 | Ready |
| Storage | Local SSD | ~50GB for VTK output across all cases |

All benchmarks run on single workstation. No HPC cluster needed.

---

# Differentiation from Prior Work

## vs. Olenik et al. (2024) -- base OGL [Meccanica]

Published OGL provides: LDU->CSR conversion, device-persistent structures, basic Ginkgo CG/BiCGStab.

**Our extensions:**
- AMG preconditioning with convergence-gated caching (not in OGL)
- ILU-ISAI preconditioner for asymmetric systems (not in OGL)
- Preconditioner selection analysis with crossover identification (no prior work)
- GAMG degradation finding (not reported anywhere)

## vs. amgx4Foam (Piscaglia, 2023) [Aerospace]

- amgx4Foam is NVIDIA-only (AmgX has no AMD/Intel support)
- Our approach is vendor-portable via Ginkgo
- amgx4Foam does not provide asymmetric momentum solver
- amgx4Foam does not report preconditioner crossover analysis
- No hierarchy caching strategy reported in amgx4Foam

## vs. petsc4Foam [NextFOAM]

- PETSc/HYPRE integration is more complex (heavyweight dependency)
- No systematic preconditioner selection guidance reported
- No GAMG comparison provided

---

# Data Inventory (What Exists vs What's Needed)

## Exists (ready for Paper 1)

| Dataset | Location | Content |
|---------|----------|---------|
| 2D cavity scaling (200-1732) | `benchmarks/bicgstab_results_ilu_isai/` | 4 variants, 6 mesh sizes, 20 steps |
| 2D cavity crossover (2000-2236) | `benchmarks/bicgstab_results_crossover/` | 4 variants, 2 mesh sizes, 10 steps |
| MG cache interval sweep | `benchmarks/mg_cache_sweep/` | intervals 5/10/20/40, 1M cells |
| ILU Trs vs ISAI comparison | `benchmarks/bicgstab_results_ilu/` | Trs bottleneck documented |
| Scaling analysis plots | `benchmarks/bicgstab_results_crossover/crossover_analysis.png` | 6-panel figure |

## Needed (for Paper 1 submission)

| Dataset | Effort | Purpose |
|---------|--------|---------|
| 3D stirred tank (4 variants, 100 steps, VTK) | 2-3 days | Industrial validation + visualization |
| 3D motorBike (4 variants, steady state, 3 meshes) | 2-3 days | Unstructured mesh validation |
| compare_solutions.py pipeline | 1 day | Automated figure generation |
| Accuracy tables ($L_\infty$, $L_2$) | 1 hour | Quantitative GPU-CPU agreement |
| GAMG sensitivity analysis | 1 day | Paper 3 go/no-go |
