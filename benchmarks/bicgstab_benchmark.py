#!/usr/bin/env python3
"""
Benchmark: GPU BiCGStab momentum solver vs CPU baseline.

Compares three variants on the cavity case at different mesh sizes:
  1. cpu       — Original OpenFOAM (smoothSolver+GS for U, GAMG for p)
  2. gpu_p     — GPU pressure only (OGLPCG for p, smoothSolver for U)
  3. gpu_all   — GPU everything (OGLPCG for p, OGLBiCGStab for U/k/epsilon)

Usage:
    python3 bicgstab_benchmark.py              # Default: 200x200 mesh, 20 steps
    python3 bicgstab_benchmark.py --sizes 100 200 400  # Multiple mesh sizes
    python3 bicgstab_benchmark.py --steps 50   # More timesteps
"""
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

CONTAINER = "ogl-dev"
# Local (host) directory for case files — mounted into container at /workspace/benchmarks
LOCAL_DIR = Path(__file__).parent / "bicgstab_cases"
# Container-side path to the same directory
CONTAINER_DIR = "/workspace/benchmarks/bicgstab_cases"


@dataclass
class RunResult:
    variant: str
    mesh_size: int
    num_cells: int = 0
    num_steps: int = 0
    total_time: float = 0.0
    avg_step_time: float = 0.0
    step_times: list = field(default_factory=list)
    # Per-field solver stats: {field: {iters: [...], times_ms: [...]}}
    solver_stats: dict = field(default_factory=dict)
    error: str = ""


def docker_exec(cmd: str, timeout: int = 300) -> str:
    """Run a command inside the dev container with OpenFOAM env sourced."""
    script = (
        'export FOAM_INST_DIR=/opt\n'
        'source /opt/OpenFOAM-13/etc/bashrc\n'
        'export GINKGO_ROOT=/opt/ginkgo\n'
        + cmd + '\n'
    )
    result = subprocess.run(
        ['docker', 'exec', '-i', CONTAINER, 'bash'],
        input=script, capture_output=True, text=True, timeout=timeout
    )
    return result.stdout + result.stderr


# --- fvSolution templates ---

FVSOLUTION_CPU = '''FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}}

solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }}

    pFinal
    {{
        $p;
        tolerance       1e-06;
        relTol          0;
    }}

    "(U|k|epsilon|omega|nuTilda).*"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }}
}}

PIMPLE
{{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}}
'''

FVSOLUTION_GPU_P = '''FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}}

solvers
{{
    p
    {{
        solver          OGLPCG;
        tolerance       1e-06;
        relTol          0.1;
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      blockJacobi;
            blockSize           4;
            isaiSparsityPower   1;
        }}
    }}

    pFinal
    {{
        solver          OGLPCG;
        tolerance       1e-06;
        relTol          0;
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      blockJacobi;
            blockSize           4;
            isaiSparsityPower   1;
        }}
    }}

    "(U|k|epsilon|omega|nuTilda).*"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }}
}}

PIMPLE
{{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}}
'''

FVSOLUTION_GPU_ALL = '''FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}}

solvers
{{
    p
    {{
        solver          OGLPCG;
        tolerance       1e-06;
        relTol          0.1;
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      blockJacobi;
            blockSize           4;
            isaiSparsityPower   1;
        }}
    }}

    pFinal
    {{
        solver          OGLPCG;
        tolerance       1e-06;
        relTol          0;
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      blockJacobi;
            blockSize           4;
            isaiSparsityPower   1;
        }}
    }}

    "(U|k|epsilon|omega|nuTilda)"
    {{
        solver          OGLBiCGStab;
        tolerance       1e-05;
        relTol          0.1;
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      blockJacobi;
            blockSize           4;
        }}
    }}

    "(U|k|epsilon|omega|nuTilda)Final"
    {{
        solver          OGLBiCGStab;
        tolerance       1e-05;
        relTol          0;
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      blockJacobi;
            blockSize           4;
        }}
    }}
}}

PIMPLE
{{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}}
'''

OGL_MG_PRESSURE_BLOCK = '''        solver          OGLPCG;
        tolerance       1e-06;
        relTol          {relTol};
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      multigrid;
            blockSize           4;
            isaiSparsityPower   1;
            mgMaxLevels         10;
            mgMinCoarseRows     64;
            mgSmootherIters     2;
            mgSmootherRelax     0.9;
            mgSmoother          chebyshev;
            mgCacheInterval     {mgCacheInterval};
            mgCacheMaxIters     200;
        }}'''

OGL_MG_PRESSURE_MIXED_BLOCK = '''        solver          OGLPCG;
        tolerance       1e-06;
        relTol          {relTol};
        OGLCoeffs
        {{
            precisionPolicy     FP32;
            iterativeRefinement on;
            maxRefineIters      3;
            innerTolerance      1e-4;
            cacheStructure      true;
            cacheValues         false;
            debug               {debug};
            preconditioner      multigrid;
            blockSize           4;
            isaiSparsityPower   1;
            mgMaxLevels         10;
            mgMinCoarseRows     64;
            mgSmootherIters     2;
            mgSmootherRelax     0.9;
            mgSmoother          chebyshev;
            mgCacheInterval     5;
            mgCacheMaxIters     200;
        }}'''

OGL_BICGSTAB_MIXED_BLOCK = '''        solver          OGLBiCGStab;
        tolerance       1e-05;
        relTol          {relTol};
        OGLCoeffs
        {{
            precisionPolicy     FP32;
            iterativeRefinement on;
            maxRefineIters      3;
            innerTolerance      1e-4;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      blockJacobi;
            blockSize           4;
        }}'''

OGL_BICGSTAB_ILU_BLOCK = '''        solver          OGLBiCGStab;
        tolerance       1e-05;
        relTol          {relTol};
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      ILU;
        }}'''

FVSOLUTION_GPU_MG_ILU = '''FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}}

solvers
{{
    p
    {{
{p_block}
    }}

    pFinal
    {{
{pf_block}
    }}

    "(U|k|epsilon|omega|nuTilda)"
    {{
{u_block}
    }}

    "(U|k|epsilon|omega|nuTilda)Final"
    {{
{uf_block}
    }}
}}

PIMPLE
{{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}}
'''

FVSOLUTION_GPU_MG_MIXED = '''FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}}

solvers
{{
    p
    {{
{p_block}
    }}

    pFinal
    {{
{pf_block}
    }}

    "(U|k|epsilon|omega|nuTilda).*"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }}
}}

PIMPLE
{{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}}
'''

FVSOLUTION_GPU_MG_ALL_MIXED = '''FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}}

solvers
{{
    p
    {{
{p_block}
    }}

    pFinal
    {{
{pf_block}
    }}

    "(U|k|epsilon|omega|nuTilda)"
    {{
{u_block}
    }}

    "(U|k|epsilon|omega|nuTilda)Final"
    {{
{uf_block}
    }}
}}

PIMPLE
{{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}}
'''

FVSOLUTION_GPU_MG = '''FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}}

solvers
{{
    p
    {{
{p_block}
    }}

    pFinal
    {{
{pf_block}
    }}

    "(U|k|epsilon|omega|nuTilda).*"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }}
}}

PIMPLE
{{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}}
'''

FVSOLUTION_GPU_MG_ALL = '''FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}}

solvers
{{
    p
    {{
{p_block}
    }}

    pFinal
    {{
{pf_block}
    }}

    "(U|k|epsilon|omega|nuTilda)"
    {{
        solver          OGLBiCGStab;
        tolerance       1e-05;
        relTol          0.1;
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      blockJacobi;
            blockSize           4;
        }}
    }}

    "(U|k|epsilon|omega|nuTilda)Final"
    {{
        solver          OGLBiCGStab;
        tolerance       1e-05;
        relTol          0;
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      blockJacobi;
            blockSize           4;
        }}
    }}
}}

PIMPLE
{{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}}
'''

CONTROLDICT_TEMPLATE = '''FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}
{libs_line}
solver          incompressibleFluid;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          {delta_t};

writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      2;
writeFormat     binary;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
'''

BLOCKMESH_TEMPLATE = '''FoamFile
{{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 0.1;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 0.1)
    (1 0 0.1)
    (1 1 0.1)
    (0 1 0.1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({n} {n} 1) simpleGrading (1 1 1)
);

boundary
(
    movingWall
    {{
        type wall;
        faces
        (
            (3 7 6 2)
        );
    }}
    fixedWalls
    {{
        type wall;
        faces
        (
            (0 4 7 3)
            (2 6 5 1)
            (1 5 4 0)
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);
'''


def prepare_case(mesh_n: int, variant: str, num_steps: int,
                  mg_cache_interval: int = 5) -> str:
    """Prepare a cavity case variant locally (host-side). Returns container path."""
    import shutil

    local_case = LOCAL_DIR / f"{variant}_{mesh_n}"
    container_case = f"{CONTAINER_DIR}/{variant}_{mesh_n}"

    # Copy base case from the repo tutorials
    src = Path(__file__).parent.parent / "tutorials" / "incompressibleFluid" / "cavity"
    if local_case.exists():
        shutil.rmtree(local_case)
    shutil.copytree(src, local_case, symlinks=True)

    # Write blockMeshDict with scaled mesh
    (local_case / "system" / "blockMeshDict").write_text(
        BLOCKMESH_TEMPLATE.format(n=mesh_n)
    )

    # Write fvSolution
    if variant == "cpu":
        fvsol = FVSOLUTION_CPU.format()
    elif variant == "gpu_p":
        fvsol = FVSOLUTION_GPU_P.format()
    elif variant == "gpu_all":
        fvsol = FVSOLUTION_GPU_ALL.format()
    elif variant == "gpu_mg":
        p_block = OGL_MG_PRESSURE_BLOCK.format(relTol=0.1, mgCacheInterval=mg_cache_interval)
        pf_block = OGL_MG_PRESSURE_BLOCK.format(relTol=0, mgCacheInterval=mg_cache_interval)
        fvsol = FVSOLUTION_GPU_MG.format(p_block=p_block, pf_block=pf_block)
    elif variant == "gpu_mg_all":
        p_block = OGL_MG_PRESSURE_BLOCK.format(relTol=0.1, mgCacheInterval=mg_cache_interval)
        pf_block = OGL_MG_PRESSURE_BLOCK.format(relTol=0, mgCacheInterval=mg_cache_interval)
        fvsol = FVSOLUTION_GPU_MG_ALL.format(p_block=p_block, pf_block=pf_block)
    elif variant == "gpu_mg_ilu":
        p_block = OGL_MG_PRESSURE_BLOCK.format(relTol=0.1, mgCacheInterval=mg_cache_interval)
        pf_block = OGL_MG_PRESSURE_BLOCK.format(relTol=0, mgCacheInterval=mg_cache_interval)
        u_block = OGL_BICGSTAB_ILU_BLOCK.format(relTol=0.1)
        uf_block = OGL_BICGSTAB_ILU_BLOCK.format(relTol=0)
        fvsol = FVSOLUTION_GPU_MG_ILU.format(
            p_block=p_block, pf_block=pf_block,
            u_block=u_block, uf_block=uf_block,
        )
    elif variant == "gpu_mg_mixed":
        p_block = OGL_MG_PRESSURE_MIXED_BLOCK.format(relTol=0.1, debug=0)
        pf_block = OGL_MG_PRESSURE_MIXED_BLOCK.format(relTol=0, debug=0)
        fvsol = FVSOLUTION_GPU_MG_MIXED.format(p_block=p_block, pf_block=pf_block)
    elif variant == "gpu_mg_all_mixed":
        p_block = OGL_MG_PRESSURE_MIXED_BLOCK.format(relTol=0.1, debug=0)
        pf_block = OGL_MG_PRESSURE_MIXED_BLOCK.format(relTol=0, debug=0)
        u_block = OGL_BICGSTAB_MIXED_BLOCK.format(relTol=0.1)
        uf_block = OGL_BICGSTAB_MIXED_BLOCK.format(relTol=0)
        fvsol = FVSOLUTION_GPU_MG_ALL_MIXED.format(
            p_block=p_block, pf_block=pf_block,
            u_block=u_block, uf_block=uf_block,
        )
    else:
        fvsol = FVSOLUTION_CPU.format()
    (local_case / "system" / "fvSolution").write_text(fvsol)

    # Write controlDict
    dt = 0.005 * (20.0 / mesh_n)
    end_time = num_steps * dt
    libs_line = 'libs ("libOGL.so");\n' if variant.startswith("gpu") else ""
    ctrldict = CONTROLDICT_TEMPLATE.format(
        libs_line=libs_line,
        end_time=end_time,
        delta_t=dt,
        write_interval=max(1, num_steps // 2),
    )
    (local_case / "system" / "controlDict").write_text(ctrldict)

    return container_case


def run_case(case_dir: str, variant: str, mesh_n: int,
             timeout: int = 600) -> RunResult:
    """Run blockMesh + foamRun and parse the log."""
    result = RunResult(variant=variant, mesh_size=mesh_n)

    # Clean any previous run data
    docker_exec(f"cd {case_dir} && rm -rf 0.* [1-9]* processor* log.*")

    # blockMesh
    out = docker_exec(f"cd {case_dir} && blockMesh 2>&1", timeout=120)
    # Try multiple patterns for cell count
    cells_match = (
        re.search(r'nCells:\s*(\d+)', out)
        or re.search(r'cells:\s*(\d+)', out)
        or re.search(r'Cell count\s*:\s*(\d+)', out, re.IGNORECASE)
    )
    if cells_match:
        result.num_cells = int(cells_match.group(1))
    else:
        # Fallback: use mesh_n^2 for 2D cavity
        result.num_cells = mesh_n * mesh_n

    # foamRun
    out = docker_exec(f"cd {case_dir} && foamRun 2>&1", timeout=timeout)

    # Parse timing
    step_times = []
    for m in re.finditer(r'ExecutionTime = ([\d.]+) s', out):
        step_times.append(float(m.group(1)))

    if len(step_times) >= 2:
        # step_times are cumulative; convert to per-step
        per_step = []
        for i in range(1, len(step_times)):
            per_step.append(step_times[i] - step_times[i - 1])
        result.step_times = per_step
        result.num_steps = len(per_step)
        result.total_time = step_times[-1] - step_times[0]
        result.avg_step_time = result.total_time / result.num_steps if result.num_steps > 0 else 0
    elif len(step_times) == 1:
        result.step_times = [step_times[0]]
        result.num_steps = 1
        result.total_time = step_times[0]
        result.avg_step_time = step_times[0]

    # Parse solver stats per field
    solver_stats = {}
    for m in re.finditer(
        r'(?:OGLBiCGStab|OGLPCG|smoothSolver|GAMG):?(?:FP\d+:)?\s+Solving for (\w+),'
        r' Initial residual = ([\d.eE+-]+),'
        r' Final residual = ([\d.eE+-]+),'
        r' No Iterations (\d+)',
        out
    ):
        field_name = m.group(1)
        iters = int(m.group(4))
        if field_name not in solver_stats:
            solver_stats[field_name] = {'iters': [], 'init_res': [], 'final_res': []}
        solver_stats[field_name]['iters'].append(iters)
        solver_stats[field_name]['init_res'].append(float(m.group(2)))
        solver_stats[field_name]['final_res'].append(float(m.group(3)))
    result.solver_stats = solver_stats

    # Check for errors
    if 'FOAM FATAL' in out:
        err_match = re.search(r'FOAM FATAL.*?(?=\n\n|\Z)', out, re.DOTALL)
        result.error = err_match.group(0)[:200] if err_match else "FATAL ERROR"

    return result


def print_results(results: list[RunResult]):
    """Print a comparison table."""
    print("\n" + "=" * 80)
    print(f"{'Variant':<12} {'Mesh':>8} {'Cells':>8} {'Steps':>6} "
          f"{'Total(s)':>10} {'Avg/step(ms)':>14} {'Speedup':>8}")
    print("-" * 80)

    # Group by mesh size
    by_mesh = {}
    for r in results:
        by_mesh.setdefault(r.mesh_size, []).append(r)

    for mesh_n in sorted(by_mesh.keys()):
        runs = by_mesh[mesh_n]
        cpu_time = None
        for r in runs:
            if r.variant == "cpu":
                cpu_time = r.avg_step_time
        for r in sorted(runs, key=lambda x: x.variant):
            speedup = cpu_time / r.avg_step_time if cpu_time and r.avg_step_time > 0 else 0
            err = f"  ERR: {r.error[:40]}" if r.error else ""
            print(
                f"{r.variant:<12} {r.mesh_size:>6}x{r.mesh_size:<1} {r.num_cells:>8} "
                f"{r.num_steps:>6} {r.total_time:>10.3f} "
                f"{r.avg_step_time * 1000:>14.2f} {speedup:>7.2f}x{err}"
            )
        print()

    # Solver iteration comparison
    print("\nSolver iterations (avg per solve):")
    print(f"{'Variant':<12} {'Mesh':>8} ", end="")
    all_fields = set()
    for r in results:
        all_fields.update(r.solver_stats.keys())
    for f in sorted(all_fields):
        print(f"{f:>10}", end="")
    print()
    print("-" * (22 + 10 * len(all_fields)))

    for mesh_n in sorted(by_mesh.keys()):
        for r in sorted(by_mesh[mesh_n], key=lambda x: x.variant):
            print(f"{r.variant:<12} {r.mesh_size:>6}x{r.mesh_size:<1} ", end="")
            for f in sorted(all_fields):
                if f in r.solver_stats and r.solver_stats[f]['iters']:
                    avg = sum(r.solver_stats[f]['iters']) / len(r.solver_stats[f]['iters'])
                    print(f"{avg:>10.1f}", end="")
                else:
                    print(f"{'—':>10}", end="")
            print()
        print()


def plot_results(results: list[RunResult], output_dir: str):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Group by mesh size
    by_mesh = {}
    for r in results:
        by_mesh.setdefault(r.mesh_size, []).append(r)

    # Determine which variants are present
    variant_order = [
        'cpu', 'gpu_p', 'gpu_all', 'gpu_mg', 'gpu_mg_all', 'gpu_mg_ilu',
        'gpu_mg_mixed', 'gpu_mg_all_mixed',
    ]
    variants = sorted(set(r.variant for r in results),
                      key=lambda v: variant_order.index(v)
                      if v in variant_order else 99)
    colors = {
        'cpu': '#2196F3', 'gpu_p': '#FF9800', 'gpu_all': '#4CAF50',
        'gpu_mg': '#9C27B0', 'gpu_mg_all': '#E91E63', 'gpu_mg_ilu': '#4CAF50',
        'gpu_mg_mixed': '#00BCD4', 'gpu_mg_all_mixed': '#FF5722',
    }
    labels = {
        'cpu': 'CPU (GAMG+GS)',
        'gpu_p': 'GPU-p (PCG+BJ)',
        'gpu_all': 'GPU-all (PCG+BJ)',
        'gpu_mg': 'GPU-p (MG-PCG FP64)',
        'gpu_mg_all': 'GPU-all (MG+BiCG+BJ)',
        'gpu_mg_ilu': 'GPU-all (MG+BiCG+ILU)',
        'gpu_mg_mixed': 'GPU-p (MG-PCG FP32)',
        'gpu_mg_all_mixed': 'GPU-all (MG+BiCG FP32)',
    }

    mesh_sizes = sorted(by_mesh.keys())

    # ---- Plot 1: Avg step time vs mesh size (bar chart) ----
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(mesh_sizes))
    nv = len(variants)
    width = 0.8 / max(nv, 1)

    for i, v in enumerate(variants):
        times = []
        for ms in mesh_sizes:
            r = next((r for r in by_mesh[ms] if r.variant == v), None)
            times.append(r.avg_step_time * 1000 if r and r.avg_step_time > 0 else 0)
        bars = ax.bar(x + i * width, times, width,
                      label=labels.get(v, v), color=colors.get(v, 'gray'))
        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{t:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Mesh Size (NxN)')
    ax.set_ylabel('Avg Step Time (ms)')
    ax.set_title('Cavity Benchmark: Step Time Comparison')
    ax.set_xticks(x + width * nv / 2)
    ax.set_xticklabels([f'{n}x{n}\n({n*n} cells)' for n in mesh_sizes])
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/step_time_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved {output_dir}/step_time_comparison.png")

    # ---- Plot 2: Speedup vs mesh size ----
    fig, ax = plt.subplots(figsize=(10, 6))

    for v in [x for x in variants if x != 'cpu']:
        speedups = []
        for ms in mesh_sizes:
            cpu_r = next((r for r in by_mesh[ms] if r.variant == 'cpu'), None)
            gpu_r = next((r for r in by_mesh[ms] if r.variant == v), None)
            if cpu_r and gpu_r and gpu_r.avg_step_time > 0:
                speedups.append(cpu_r.avg_step_time / gpu_r.avg_step_time)
            else:
                speedups.append(0)
        ax.plot(mesh_sizes, speedups, 'o-', color=colors.get(v, 'gray'),
                label=labels.get(v, v), linewidth=2, markersize=8)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlabel('Mesh Size (N)')
    ax.set_ylabel('Speedup vs CPU')
    ax.set_title('Cavity Benchmark: GPU Speedup')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_vs_mesh.png', dpi=150)
    plt.close()
    print(f"  Saved {output_dir}/speedup_vs_mesh.png")

    # ---- Plot 3: Per-step timeline for largest mesh ----
    largest = max(mesh_sizes)
    fig, ax = plt.subplots(figsize=(12, 5))
    for v in variants:
        r = next((r for r in by_mesh[largest] if r.variant == v), None)
        if r and r.step_times:
            times_ms = [t * 1000 for t in r.step_times]
            steps = list(range(1, len(times_ms) + 1))
            ax.plot(steps, times_ms, '-', color=colors[v],
                    label=labels[v], linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Step Time (ms)')
    ax.set_title(f'Per-Step Timing ({largest}x{largest} cavity, {largest*largest} cells)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_step_timeline.png', dpi=150)
    plt.close()
    print(f"  Saved {output_dir}/per_step_timeline.png")

    # ---- Plot 4: Solver iterations comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, field_name in enumerate(['Ux', 'p']):
        ax = axes[ax_idx]
        for v in variants:
            r = next((r for r in by_mesh[largest] if r.variant == v), None)
            if r and field_name in r.solver_stats:
                iters = r.solver_stats[field_name]['iters']
                ax.plot(range(1, len(iters) + 1), iters, '-', color=colors[v],
                        label=f'{labels[v]} (avg={sum(iters)/len(iters):.1f})',
                        linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Solve Index')
        ax.set_ylabel('Iterations')
        ax.set_title(f'{field_name} Solver Iterations ({largest}x{largest})')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/solver_iterations.png', dpi=150)
    plt.close()
    print(f"  Saved {output_dir}/solver_iterations.png")


def main():
    parser = argparse.ArgumentParser(description='BiCGStab momentum benchmark')
    parser.add_argument('--sizes', nargs='+', type=int, default=[100, 200, 400],
                        help='Mesh sizes (NxN)')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of timesteps')
    parser.add_argument('--output', type=str, default='bicgstab_results',
                        help='Output directory for results and plots')
    parser.add_argument('--variants', type=str,
                        default='cpu,gpu_mg,gpu_mg_all',
                        help='Variants: cpu,gpu_p,gpu_all,gpu_mg,gpu_mg_all,gpu_mg_ilu,gpu_mg_mixed,gpu_mg_all_mixed')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Timeout in seconds for foamRun (increase for large meshes)')
    parser.add_argument('--mg-cache-interval', type=int, default=5,
                        help='MG hierarchy rebuild interval (default 5)')
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    variants = args.variants.split(',')
    all_results = []

    for mesh_n in args.sizes:
        print(f"\n{'='*60}")
        print(f"Mesh: {mesh_n}x{mesh_n} ({mesh_n*mesh_n} cells), {args.steps} steps")
        print(f"{'='*60}")

        for variant in variants:
            print(f"\n  [{variant}] Preparing case...")
            case_dir = prepare_case(mesh_n, variant, args.steps,
                                    mg_cache_interval=args.mg_cache_interval)

            print(f"  [{variant}] Running (timeout={args.timeout}s)...")
            result = run_case(case_dir, variant, mesh_n, timeout=args.timeout)
            all_results.append(result)

            if result.error:
                print(f"  [{variant}] ERROR: {result.error[:80]}")
            else:
                print(f"  [{variant}] Done: {result.num_steps} steps, "
                      f"avg {result.avg_step_time*1000:.1f} ms/step")

    # Print summary table
    print_results(all_results)

    # Save results with metadata
    from datetime import datetime
    gpu_info = docker_exec("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown'").strip()
    cpu_info = docker_exec("lscpu | grep 'Model name' | sed 's/Model name: *//' 2>/dev/null || echo 'unknown'").strip()

    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'benchmark': 'bicgstab_momentum',
            'case': 'cavity (2D lid-driven, kEpsilon)',
            'gpu': gpu_info,
            'cpu': cpu_info,
            'mesh_sizes': args.sizes,
            'steps_per_size': args.steps,
            'variants': variants,
            'solver_config': {
                'cpu': 'GAMG(p) + smoothSolver+symGS(U)',
                'gpu_mg': 'OGLPCG+MG-FP64(p) + smoothSolver+symGS(U)',
                'gpu_mg_all': 'OGLPCG+MG-FP64(p) + OGLBiCGStab+BJ-FP64(U/k/eps)',
                'gpu_mg_ilu': 'OGLPCG+MG-FP64(p) + OGLBiCGStab+ILU-FP64(U/k/eps)',
                'gpu_mg_mixed': 'OGLPCG+MG-FP32+refine(p) + smoothSolver+symGS(U)',
                'gpu_mg_all_mixed': 'OGLPCG+MG-FP32+refine(p) + OGLBiCGStab+BJ-FP32+refine(U/k/eps)',
            },
        },
        'results': [],
    }
    for r in all_results:
        output['results'].append({
            'variant': r.variant,
            'mesh_size': r.mesh_size,
            'num_cells': r.num_cells,
            'num_steps': r.num_steps,
            'total_time': r.total_time,
            'avg_step_time': r.avg_step_time,
            'step_times': r.step_times,
            'solver_stats': r.solver_stats,
            'error': r.error,
        })
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")

    # Generate plots
    print("\nGenerating plots...")
    plot_results(all_results, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
