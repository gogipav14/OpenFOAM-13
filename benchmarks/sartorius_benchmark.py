#!/usr/bin/env python3
"""
Benchmark: Sartorius 50L Palletank mixing case — CPU vs GPU solver comparison.

Runs the 50L Palletank case at multiple RPMs with CPU and GPU solver variants.
Measures wall time, iteration counts, and tracer CoV for mixing time estimation.

Usage:
    python3 sartorius_benchmark.py                     # Default: 200 RPM, 20 steps
    python3 sartorius_benchmark.py --rpms 100 200 300  # Multiple RPMs
    python3 sartorius_benchmark.py --steps 50          # More timesteps
    python3 sartorius_benchmark.py --variants cpu gpu   # Specific variants
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

CONTAINER = "ogl-dev"
# Source case (template) — use mixing_cases/ as canonical ASCII source
SOURCE_CASE = Path(__file__).parent.parent / "mixing_cases" / "sartorius_50L_benchmark"
# Working directory for benchmark runs
WORK_DIR = Path(__file__).parent / "sartorius_runs"
CONTAINER_WORK = "/workspace/benchmarks/sartorius_runs"


# Base mesh: 40x40x46 = 73,600 cells
BASE_NX, BASE_NY, BASE_NZ = 40, 40, 46

# Vessel dimensions
VESSEL_L, VESSEL_W, VESSEL_H = 0.41, 0.41, 0.465
IMPELLER_D = 0.126


@dataclass
class RunResult:
    variant: str
    rpm: int
    mesh_tag: str = ""
    num_cells: int = 0
    num_steps: int = 0
    total_time: float = 0.0
    avg_step_time: float = 0.0
    step_times: list = field(default_factory=list)
    solver_stats: dict = field(default_factory=dict)
    tracer_cov: list = field(default_factory=list)
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

FVSOLUTION_CPU = """\
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}

solvers
{{
    p
    {{
        solver          GAMG;
        smoother        DIC;
        tolerance       1e-6;
        relTol          0.01;
    }}

    pFinal
    {{
        $p;
        relTol          0;
    }}

    "(U|k|epsilon)"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }}

    "(U|k|epsilon)Final"
    {{
        $U;
        relTol          0;
    }}

    tracer
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;
    }}
}}

PIMPLE
{{
    momentumPredictor yes;
    nOuterCorrectors  2;
    nCorrectors       2;
    nNonOrthogonalCorrectors 0;
    pRefCell          0;
    pRefValue         0;
}}

relaxationFactors
{{
    equations
    {{
        ".*"            1;
    }}
}}
"""

OGL_PRESSURE_BLOCK = """\
    p
    {{
        solver          OGLPCG;
        tolerance       1e-6;
        relTol          0.01;
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
            mgCacheInterval     10;
            mgCacheMaxIters     200;
        }}
    }}

    pFinal
    {{
        solver          OGLPCG;
        tolerance       1e-6;
        relTol          0;
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
            mgCacheInterval     10;
            mgCacheMaxIters     200;
        }}
    }}
"""

CPU_MOMENTUM_BLOCK = """\
    "(U|k|epsilon)"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }}

    "(U|k|epsilon)Final"
    {{
        $U;
        relTol          0;
    }}
"""

GPU_BJ_MOMENTUM_BLOCK = """\
    "(U|k|epsilon)"
    {{
        solver          OGLBiCGStab;
        tolerance       1e-6;
        relTol          0.1;
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      blockJacobi;
        }}
    }}

    "(U|k|epsilon)Final"
    {{
        solver          OGLBiCGStab;
        tolerance       1e-6;
        relTol          0;
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      blockJacobi;
        }}
    }}
"""

GPU_ILU_MOMENTUM_BLOCK = """\
    "(U|k|epsilon)"
    {{
        solver          OGLBiCGStab;
        tolerance       1e-6;
        relTol          0.1;
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      ILU;
        }}
    }}

    "(U|k|epsilon)Final"
    {{
        solver          OGLBiCGStab;
        tolerance       1e-6;
        relTol          0;
        OGLCoeffs
        {{
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      ILU;
        }}
    }}
"""

TRACER_BLOCK = """\
    tracer
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;
    }}
"""

PIMPLE_BLOCK = """\
PIMPLE
{{
    momentumPredictor yes;
    nOuterCorrectors  2;
    nCorrectors       2;
    nNonOrthogonalCorrectors 0;
    pRefCell          0;
    pRefValue         0;
}}

relaxationFactors
{{
    equations
    {{
        ".*"            1;
    }}
}}
"""

# Variant definitions:
#   cpu     = GAMG + symGS (vanilla OpenFOAM)
#   gpu_p   = GPU AMG-PCG pressure + CPU symGS momentum
#   gpu_bj  = GPU AMG-PCG pressure + GPU BiCGStab+BJ momentum
#   gpu_ilu = GPU AMG-PCG pressure + GPU BiCGStab+ILU+ISAI momentum
VARIANT_LABELS = {
    'cpu':     'CPU (GAMG+symGS)',
    'gpu_p':   'GPU-p (AMG-PCG)',
    'gpu_bj':  'GPU-all (BJ)',
    'gpu_ilu': 'GPU-all (ILU-ISAI)',
}


def _unesc(s: str) -> str:
    """Convert double-braces to single (undo f-string/format escaping)."""
    return s.replace("{{", "{").replace("}}", "}")


def build_fvsolution(variant: str) -> str:
    """Build fvSolution content for a given variant."""
    header = "FoamFile\n{\n    format      ascii;\n    class       dictionary;\n    object      fvSolution;\n}\n\nsolvers\n{\n"
    footer = "}\n\n"

    if variant == "cpu":
        pressure = """\
    p
    {
        solver          GAMG;
        smoother        DIC;
        tolerance       1e-6;
        relTol          0.01;
    }

    pFinal
    {
        $p;
        relTol          0;
    }
"""
        momentum = _unesc(CPU_MOMENTUM_BLOCK)
    elif variant == "gpu_p":
        pressure = _unesc(OGL_PRESSURE_BLOCK)
        momentum = _unesc(CPU_MOMENTUM_BLOCK)
    elif variant == "gpu_bj":
        pressure = _unesc(OGL_PRESSURE_BLOCK)
        momentum = _unesc(GPU_BJ_MOMENTUM_BLOCK)
    elif variant == "gpu_ilu":
        pressure = _unesc(OGL_PRESSURE_BLOCK)
        momentum = _unesc(GPU_ILU_MOMENTUM_BLOCK)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return (header + pressure + momentum
            + _unesc(TRACER_BLOCK) + footer + _unesc(PIMPLE_BLOCK))

CONTROLDICT_TEMPLATE = """\
FoamFile
{{
    format      ascii;
    class       dictionary;
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
purgeWrite      3;

writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable no;

functions
{{
    tracerTransport
    {{
        type            scalarTransport;
        libs            ("libsolverFunctionObjects.so");
        field           tracer;
        diffusivity     viscosity;
        alphal          1;
        alphat          1;
    }}

    mixingQuality
    {{
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        log             true;
        operation       CoV;
        fields          (tracer);
        cellZone        all;
    }}
}}
"""


def prepare_case(rpm: int, variant: str, num_steps: int,
                 mesh_factor: float = 1.0) -> tuple[str, str]:
    """Prepare a case directory for a specific RPM, variant, and mesh size.

    Returns (container_case_path, mesh_tag).
    """
    # Scale mesh
    nx = int(BASE_NX * mesh_factor)
    ny = int(BASE_NY * mesh_factor)
    nz = int(BASE_NZ * mesh_factor)
    approx_cells = nx * ny * nz
    mesh_tag = f"{approx_cells // 1000}K" if approx_cells >= 1000 else str(approx_cells)

    case_name = f"{variant}_rpm{rpm}_{mesh_tag}"
    local_case = WORK_DIR / case_name
    container_case = f"{CONTAINER_WORK}/{case_name}"

    # Clean previous
    if local_case.exists():
        try:
            shutil.rmtree(local_case)
        except PermissionError:
            docker_exec(f"rm -rf {container_case}", timeout=30)
            if local_case.exists():
                shutil.rmtree(local_case, ignore_errors=True)

    # Copy template case
    shutil.copytree(SOURCE_CASE, local_case, dirs_exist_ok=False,
                    ignore=shutil.ignore_patterns('polyMesh', '0.*', '[1-9]*',
                                                  'log.*', 'postProcessing'))

    # Write scaled blockMeshDict
    blockmesh = f"""\
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

L   {VESSEL_L};
W   {VESSEL_W};
H   {VESSEL_H};

nx  {nx};
ny  {ny};
nz  {nz};

vertices
(
    (0    0    0)
    ($L   0    0)
    ($L   $W   0)
    (0    $W   0)
    (0    0    $H)
    ($L   0    $H)
    ($L   $W   $H)
    (0    $W   $H)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ($nx $ny $nz) simpleGrading (1 1 1)
);

boundary
(
    bottom
    {{
        type wall;
        faces ( (0 3 2 1) );
    }}
    top
    {{
        type patch;
        faces ( (4 5 6 7) );
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 1 5 4)
            (1 2 6 5)
            (2 3 7 6)
            (0 4 7 3)
        );
    }}
);

mergePatchPairs ();
"""
    (local_case / "system" / "blockMeshDict").write_text(blockmesh)

    # Write MRFProperties with this RPM
    mrf = f"""\
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      MRFProperties;
}}

MRF1
{{
    cellZone    rotatingZone;
    active      yes;
    nonRotatingPatches (bottom walls);
    origin      (0.205 0.205 0);
    axis        (0 0 1);
    omega       {rpm} [rpm];
}}
"""
    (local_case / "constant" / "MRFProperties").write_text(mrf)

    # Write fvSolution for this variant
    (local_case / "system" / "fvSolution").write_text(build_fvsolution(variant))

    # Write controlDict with CFL-scaled dt
    tip_speed = 3.14159 * IMPELLER_D * rpm / 60.0
    dx = VESSEL_L / nx
    dt = min(0.005, 0.7 * dx / max(tip_speed, 0.1))
    end_time = num_steps * dt
    libs_line = 'libs ("libOGL.so");' if variant.startswith("gpu") else ""

    ctrldict = CONTROLDICT_TEMPLATE.format(
        libs_line=libs_line,
        end_time=f"{end_time:.6f}",
        delta_t=f"{dt:.6f}",
        write_interval=max(1, num_steps),
    )
    (local_case / "system" / "controlDict").write_text(ctrldict)

    return container_case, mesh_tag


def run_case(case_dir: str, variant: str, rpm: int,
             mesh_tag: str = "", timeout: int = 600) -> RunResult:
    """Run blockMesh + topoSet + setFields + foamRun and parse the log."""
    result = RunResult(variant=variant, rpm=rpm, mesh_tag=mesh_tag)

    # Clean any previous run data
    docker_exec(f"cd {case_dir} && rm -rf 0.* [1-9]* constant/polyMesh log.* postProcessing")

    # blockMesh
    out = docker_exec(f"cd {case_dir} && blockMesh 2>&1", timeout=120)
    cells_match = (
        re.search(r'nCells:\s*(\d+)', out)
        or re.search(r'cells:\s*(\d+)', out)
    )
    result.num_cells = int(cells_match.group(1)) if cells_match else 73600

    # topoSet
    docker_exec(f"cd {case_dir} && topoSet 2>&1", timeout=60)

    # setFields
    docker_exec(f"cd {case_dir} && setFields 2>&1", timeout=60)

    # foamRun
    out = docker_exec(f"cd {case_dir} && foamRun 2>&1", timeout=timeout)

    # Parse timing (cumulative ExecutionTime)
    step_times_cum = []
    for m in re.finditer(r'ExecutionTime = ([\d.]+) s', out):
        step_times_cum.append(float(m.group(1)))

    if len(step_times_cum) >= 2:
        per_step = []
        for i in range(1, len(step_times_cum)):
            per_step.append(step_times_cum[i] - step_times_cum[i - 1])
        result.step_times = per_step
        result.num_steps = len(per_step)
        result.total_time = step_times_cum[-1] - step_times_cum[0]
        result.avg_step_time = result.total_time / result.num_steps if result.num_steps > 0 else 0
    elif len(step_times_cum) == 1:
        result.step_times = [step_times_cum[0]]
        result.num_steps = 1
        result.total_time = step_times_cum[0]
        result.avg_step_time = step_times_cum[0]

    # Parse solver stats
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

    # Parse tracer CoV
    for m in re.finditer(r'volFieldValue.*?CoV\(tracer\)\s*=\s*([\d.eE+-]+)', out):
        result.tracer_cov.append(float(m.group(1)))

    # Check for errors
    if 'FOAM FATAL' in out:
        err_match = re.search(r'FOAM FATAL.*?(?=\n\n|\Z)', out, re.DOTALL)
        result.error = err_match.group(0)[:300] if err_match else "FATAL ERROR"

    return result


def print_results(results: list[RunResult]):
    """Print a comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Variant':<12} {'RPM':>5} {'Mesh':>7} {'Cells':>8} {'Steps':>6} "
          f"{'Total(s)':>10} {'Avg/step(ms)':>14} {'Speedup':>8}")
    print("-" * 100)

    # Group by (rpm, mesh_tag)
    by_group = {}
    for r in results:
        key = (r.rpm, r.mesh_tag)
        by_group.setdefault(key, []).append(r)

    for key in sorted(by_group.keys()):
        rpm, mesh_tag = key
        runs = by_group[key]
        cpu_time = None
        for r in runs:
            if r.variant == "cpu" and r.total_time > 0:
                cpu_time = r.total_time

        for r in runs:
            label = VARIANT_LABELS.get(r.variant, r.variant)
            if r.error:
                print(f"{r.variant:<12} {r.rpm:>5} {r.mesh_tag:>7} {r.num_cells:>8} "
                      f"{'ERR':>6}  {r.error[:40]}")
                continue
            speedup = ""
            if cpu_time and r.total_time > 0 and r.variant != "cpu":
                speedup = f"{cpu_time / r.total_time:.2f}x"
            print(f"{r.variant:<12} {r.rpm:>5} {r.mesh_tag:>7} {r.num_cells:>8} {r.num_steps:>6} "
                  f"{r.total_time:>10.2f} {r.avg_step_time * 1000:>14.1f} {speedup:>8}")
        print()

    # Iteration count comparison
    print("\n--- Average iterations per solve ---")
    print(f"{'Variant':<12} {'Mesh':>7} ", end="")
    fields_to_show = ['p', 'Ux', 'Uy', 'Uz', 'k', 'epsilon']
    for f in fields_to_show:
        print(f"{f:>8}", end="")
    print()
    print("-" * (22 + 8 * len(fields_to_show)))

    for key in sorted(by_group.keys()):
        for r in by_group[key]:
            if r.error:
                continue
            print(f"{r.variant:<12} {r.mesh_tag:>7} ", end="")
            for f in fields_to_show:
                if f in r.solver_stats and r.solver_stats[f]['iters']:
                    avg_it = sum(r.solver_stats[f]['iters']) / len(r.solver_stats[f]['iters'])
                    print(f"{avg_it:>8.1f}", end="")
                else:
                    print(f"{'--':>8}", end="")
            print()
        print()


def save_results(results: list[RunResult], output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = []
    for r in results:
        entry = {
            'variant': r.variant,
            'rpm': r.rpm,
            'mesh_tag': r.mesh_tag,
            'num_cells': r.num_cells,
            'num_steps': r.num_steps,
            'total_time': r.total_time,
            'avg_step_time': r.avg_step_time,
            'step_times': r.step_times,
            'solver_stats': r.solver_stats,
            'tracer_cov': r.tracer_cov,
            'error': r.error,
        }
        data.append(entry)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"sartorius_benchmark_{ts}.json"
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {outfile}")

    # Also save a CSV summary
    csvfile = output_dir / f"sartorius_benchmark_{ts}.csv"
    with open(csvfile, 'w') as f:
        f.write("variant,rpm,mesh,cells,steps,total_s,avg_step_ms,speedup\n")
        by_group = {}
        for r in results:
            by_group.setdefault((r.rpm, r.mesh_tag), []).append(r)

        for key in sorted(by_group.keys()):
            rpm, mesh_tag = key
            cpu_time = None
            for r in by_group[key]:
                if r.variant == "cpu" and r.total_time > 0:
                    cpu_time = r.total_time
            for r in by_group[key]:
                speedup = ""
                if cpu_time and r.total_time > 0 and r.variant != "cpu":
                    speedup = f"{cpu_time / r.total_time:.2f}"
                f.write(f"{r.variant},{r.rpm},{r.mesh_tag},{r.num_cells},{r.num_steps},"
                        f"{r.total_time:.3f},{r.avg_step_time * 1000:.1f},{speedup}\n")
    print(f"CSV saved to {csvfile}")


def main():
    parser = argparse.ArgumentParser(description="Sartorius 50L mixing benchmark")
    parser.add_argument("--rpms", nargs="+", type=int, default=[200],
                        help="Impeller RPM values to test")
    parser.add_argument("--mesh-factors", nargs="+", type=float, default=[1.0, 2.0, 3.0],
                        help="Mesh refinement factors (1.0=74K, 2.0=~590K, 3.0=~2M)")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of timesteps per run")
    parser.add_argument("--variants", nargs="+",
                        default=["cpu", "gpu_p", "gpu_bj", "gpu_ilu"],
                        help="Solver variants: cpu, gpu_p, gpu_bj, gpu_ilu")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Timeout per run in seconds")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "sartorius_results"),
                        help="Output directory for results")
    args = parser.parse_args()

    # Verify Docker container is running
    try:
        docker_exec("echo OK", timeout=10)
    except Exception as e:
        print(f"Error: Cannot connect to Docker container '{CONTAINER}': {e}")
        sys.exit(1)

    # Verify source case exists
    if not SOURCE_CASE.exists():
        print(f"Error: Source case not found at {SOURCE_CASE}")
        sys.exit(1)

    # Create work directory
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Get hardware info
    gpu_info = docker_exec("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown'").strip()
    cpu_info = docker_exec("lscpu | grep 'Model name' | sed 's/Model name: *//' 2>/dev/null || echo 'unknown'").strip()
    print(f"GPU: {gpu_info}")
    print(f"CPU: {cpu_info}")
    print(f"RPMs: {args.rpms}")
    print(f"Mesh factors: {args.mesh_factors}")
    print(f"Variants: {args.variants}")
    print(f"Steps: {args.steps}")
    print()

    results = []
    total_runs = len(args.rpms) * len(args.mesh_factors) * len(args.variants)
    run_num = 0

    for mf in args.mesh_factors:
        for rpm in args.rpms:
            for variant in args.variants:
                run_num += 1
                nx = int(BASE_NX * mf)
                approx = nx * int(BASE_NY * mf) * int(BASE_NZ * mf)
                tag = f"{approx // 1000}K" if approx >= 1000 else str(approx)
                label = VARIANT_LABELS.get(variant, variant)
                print(f"[{run_num}/{total_runs}] {label} @ {rpm} RPM, {tag} cells ... ",
                      end="", flush=True)

                try:
                    case_dir, mesh_tag = prepare_case(rpm, variant, args.steps,
                                                      mesh_factor=mf)
                    result = run_case(case_dir, variant, rpm, mesh_tag=mesh_tag,
                                     timeout=args.timeout)

                    if result.error:
                        print(f"ERROR: {result.error[:60]}")
                    else:
                        print(f"{result.total_time:.2f}s "
                              f"({result.avg_step_time * 1000:.1f} ms/step)")

                    results.append(result)

                except subprocess.TimeoutExpired:
                    print("TIMEOUT")
                    results.append(RunResult(variant=variant, rpm=rpm,
                                            mesh_tag=tag, error="Timeout"))
                except Exception as e:
                    print(f"EXCEPTION: {e}")
                    results.append(RunResult(variant=variant, rpm=rpm,
                                            mesh_tag=tag, error=str(e)[:200]))

                # Clean up after each run to save disk space
                case_name = f"{variant}_rpm{rpm}_{tag}"
                docker_exec(f"rm -rf {CONTAINER_WORK}/{case_name}", timeout=30)

    print_results(results)
    save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
