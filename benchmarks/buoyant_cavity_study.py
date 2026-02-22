#!/usr/bin/env python3
"""
Phase 2: Variable-coefficient spectral solver validation on buoyantCavity.

The buoyantCavity case has spatially varying rAU due to buoyancy coupling
(density depends on temperature). This tests whether PCG+DCT can handle
the variable-coefficient pressure operator div(rAU*grad(p_rgh)).

Unlike the constant-coefficient cavity (Phase 1), the DCT preconditioner
is only approximate here — it inverts the constant-coefficient Laplacian
while the actual operator has variable coefficients. The PCG outer loop
converges the difference, but may require more iterations.

Base mesh: 35x150x15 = 78,750 cells (structured, blockMesh)
Domain: 76mm x 2180mm x 520mm (tall narrow cavity, hot/cold walls)

Usage:
    python buoyant_cavity_study.py --docker --image mixfoam:latest
    python buoyant_cavity_study.py --docker --variants cpu gpu_spectral
    python buoyant_cavity_study.py --docker --levels base 2x --lib-override .lib/libOGL.so
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from case_modifier import prepare_case, BenchmarkConfig
from runner import run_case_docker, RunConfig
from log_parser import parse_solver_log, compute_metrics, find_solver_log


# Domain dimensions (metres, after convertToMeters 0.001)
DOMAIN_X = 0.076   # 76 mm
DOMAIN_Y = 2.180   # 2180 mm
DOMAIN_Z = 0.520   # 520 mm

# Base mesh resolution
BASE_NX, BASE_NY, BASE_NZ = 35, 150, 15

MESH_LEVELS = [
    {
        "name": "base",
        "nx": BASE_NX, "ny": BASE_NY, "nz": BASE_NZ,
        "label": "78K (35x150x15)",
    },
    {
        "name": "2x",
        "nx": BASE_NX * 2, "ny": BASE_NY * 2, "nz": BASE_NZ * 2,
        "label": "630K (70x300x30)",
    },
]

VARIANTS = [
    {
        "name": "cpu",
        "label": "CPU (GAMG)",
        "variant": "cpu",
    },
    {
        "name": "gpu_spectral",
        "label": "GPU Spectral (PCG+DCT)",
        "variant": "gpu_spectral",
        "needs_fft_dims": True,
    },
]


def update_mesh(case_path: Path, nx: int, ny: int, nz: int) -> int:
    """Update buoyantCavity blockMeshDict to target resolution."""
    bmdict = case_path / "system" / "blockMeshDict"
    content = bmdict.read_text()

    content = re.sub(
        r'(hex\s*\([^)]+\)\s*\()\d+\s+\d+\s+\d+(\)\s*simpleGrading)',
        rf'\g<1>{nx} {ny} {nz}\g<2>',
        content
    )
    bmdict.write_text(content)
    return nx * ny * nz


def set_ncorrectors(case_path: Path, n_correctors: int):
    """Set nCorrectors in PIMPLE settings to stress the pressure solver."""
    fvsolution = case_path / "system" / "fvSolution"
    content = fvsolution.read_text()

    if re.search(r'nCorrectors\s+\d+\s*;', content):
        content = re.sub(
            r'nCorrectors\s+\d+\s*;',
            f'nCorrectors     {n_correctors};',
            content,
        )
    else:
        # Insert nCorrectors after PIMPLE { or after momentumPredictor line
        content = re.sub(
            r'(momentumPredictor\s+\w+\s*;)',
            rf'\1\n    nCorrectors     {n_correctors};',
            content,
        )
    fvsolution.write_text(content)


def run_variant(
    base_case: Path,
    level: dict,
    var: dict,
    output_dir: Path,
    bench_config: BenchmarkConfig,
    run_config: RunConfig,
    n_correctors: int = 0,
) -> dict:
    """Run a single variant at a given mesh level."""
    level_name = level["name"]
    var_name = var["name"]
    variant = var["variant"]
    nx, ny, nz = level["nx"], level["ny"], level["nz"]

    var_dir = output_dir / f"{level_name}_{var_name}"

    # Compute FFT dimensions and mesh spacing if needed
    fft_dims = None
    mesh_spacing = None
    if var.get("needs_fft_dims"):
        fft_dims = (nx, ny, nz)
        mesh_spacing = (DOMAIN_X / nx, DOMAIN_Y / ny, DOMAIN_Z / nz)

    config = BenchmarkConfig(
        max_timesteps=bench_config.max_timesteps,
        force_serial=bench_config.force_serial,
        precision_policy=bench_config.precision_policy,
        iterative_refinement=bench_config.iterative_refinement,
        max_refine_iters=bench_config.max_refine_iters,
        inner_tolerance=bench_config.inner_tolerance,
        cache_structure=bench_config.cache_structure,
        cache_values=bench_config.cache_values,
        debug_level=bench_config.debug_level,
        preconditioner=var.get("preconditioner", "Jacobi") or "Jacobi",
        block_size=var.get("block_size", 4),
        fft_dimensions=fft_dims,
        mesh_spacing=mesh_spacing,
    )

    print(f"    Preparing {var_name}...")
    prepare_case(base_case, var_dir, variant, config)

    # Update mesh if not base resolution
    if nx != BASE_NX or ny != BASE_NY or nz != BASE_NZ:
        actual_cells = update_mesh(var_dir, nx, ny, nz)
    else:
        actual_cells = nx * ny * nz

    # Set nCorrectors to increase pressure solve count per step
    if n_correctors > 0:
        set_ncorrectors(var_dir, n_correctors)

    # Force ascii write for log parsing and set writeInterval
    # CFL conditioning is handled globally by case_modifier._enforce_cfl()
    controldict = var_dir / "system" / "controlDict"
    content = controldict.read_text()
    content = re.sub(r'writeFormat\s+\w+\s*;', 'writeFormat     ascii;', content)
    content = re.sub(
        r'writeInterval\s+\d+\s*;',
        f'writeInterval   {config.max_timesteps};',
        content,
    )
    controldict.write_text(content)

    print(f"    Running {var_name} ({actual_cells:,} cells)...")

    if run_config.use_docker:
        metrics = run_case_docker(var_dir, run_config, variant=variant)
    else:
        from runner import run_case_native
        metrics = run_case_native(var_dir, run_config)

    metrics.case_name = f"buoyantCavity_{level_name}"
    metrics.variant = var_name

    result = {
        "variant": var_name,
        "label": var["label"],
        "wall_time": metrics.total_wall_time,
        "exec_time": metrics.total_exec_time,
        "time_per_step": metrics.avg_time_per_step,
        "total_iters": metrics.total_pressure_iters,
        "iters_per_step": metrics.avg_iters_per_step,
        "iters_per_solve": metrics.avg_iters_per_solve,
        "solver": metrics.solver_type,
        "num_steps": metrics.num_timesteps,
        "completed": metrics.completed,
        "error": metrics.error_message,
        "avg_final_residual": metrics.avg_final_residual,
    }

    if metrics.completed:
        print(f"    {var_name}: {metrics.avg_time_per_step:.4f} s/step, "
              f"{metrics.avg_iters_per_step:.1f} iter/step, "
              f"final_res={metrics.avg_final_residual:.2e}, "
              f"solver={metrics.solver_type}")
    else:
        print(f"    {var_name}: FAILED - {metrics.error_message}")

    # Save per-variant metrics
    metrics_path = var_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Save convergence data
    save_convergence_data(
        var_dir, output_dir / f"convergence_{level_name}_{var_name}.csv"
    )

    return result


def save_convergence_data(case_dir: Path, output_path: Path):
    """Extract per-timestep convergence data from solver log."""
    log_file = find_solver_log(case_dir)
    if not log_file:
        return

    timesteps = parse_solver_log(log_file)
    if not timesteps:
        return

    lines = [
        "step,time,exec_time,p_initial_res,p_final_res,p_iters,cumulative_iters"
    ]
    cumulative = 0
    for i, ts in enumerate(timesteps):
        if not ts.pressure_solves:
            continue
        total_iters = sum(s.iterations for s in ts.pressure_solves)
        cumulative += total_iters
        init_res = max(s.initial_residual for s in ts.pressure_solves)
        final_res = min(s.final_residual for s in ts.pressure_solves)
        lines.append(
            f"{i},{ts.time},{ts.execution_time},"
            f"{init_res},{final_res},{total_iters},{cumulative}"
        )

    output_path.write_text('\n'.join(lines) + '\n')


def run_level(
    base_case: Path,
    level: dict,
    output_dir: Path,
    bench_config: BenchmarkConfig,
    run_config: RunConfig,
    variants: list,
    n_correctors: int = 0,
) -> dict:
    """Run all variants at a given mesh level."""
    level_name = level["name"]
    nx, ny, nz = level["nx"], level["ny"], level["nz"]
    expected_cells = nx * ny * nz

    print(f"\n{'='*70}")
    print(f"  buoyantCavity: {level['label']} "
          f"({nx}x{ny}x{nz} = {expected_cells:,} cells)")
    print(f"  Variable coefficients: rAU varies due to buoyancy coupling")
    if n_correctors > 0:
        print(f"  nCorrectors: {n_correctors} (multiple pressure corrections/step)")
    print(f"{'='*70}")

    result = {
        "level": level_name,
        "cells": expected_cells,
        "nx": nx, "ny": ny, "nz": nz,
        "variants": {},
    }

    for v in variants:
        var_result = run_variant(
            base_case, level, v, output_dir, bench_config, run_config,
            n_correctors=n_correctors,
        )
        result["variants"][v["name"]] = var_result

    # Compute speedups relative to CPU
    cpu_result = result["variants"].get("cpu")
    if cpu_result and cpu_result.get("completed"):
        cpu_t = cpu_result["time_per_step"]
        for name, vr in result["variants"].items():
            if name != "cpu" and vr.get("completed") and vr["time_per_step"] > 0:
                vr["speedup_vs_cpu"] = cpu_t / vr["time_per_step"]
            else:
                vr["speedup_vs_cpu"] = 0

    return result


def print_summary(all_results: list, variants: list):
    """Print summary table."""
    print(f"\n{'='*80}")
    print("PHASE 2: VARIABLE-COEFFICIENT SPECTRAL SOLVER (buoyantCavity)")
    print(f"{'='*80}")

    # Wall time per step
    print(f"\n--- Wall-time per step [s] ---")
    header = f"{'Level':<12} {'Cells':>10}"
    for v in variants:
        header += f" | {v['label']:>26}"
    print(header)
    for r in all_results:
        line = f"{r['level']:<12} {r['cells']:>10,}"
        for v in variants:
            vr = r["variants"].get(v["name"], {})
            t = vr.get("time_per_step", 0)
            line += f" | {t:>26.4f}" if t > 0 else f" | {'FAILED':>26}"
        print(line)

    # Pressure iterations per step
    print(f"\n--- Pressure iterations per step ---")
    header = f"{'Level':<12} {'Cells':>10}"
    for v in variants:
        header += f" | {v['label']:>26}"
    print(header)
    for r in all_results:
        line = f"{r['level']:<12} {r['cells']:>10,}"
        for v in variants:
            vr = r["variants"].get(v["name"], {})
            it = vr.get("iters_per_step", 0)
            line += f" | {it:>26.1f}" if it > 0 else f" | {'FAILED':>26}"
        print(line)

    # Avg final residual
    print(f"\n--- Avg final residual ---")
    header = f"{'Level':<12} {'Cells':>10}"
    for v in variants:
        header += f" | {v['label']:>26}"
    print(header)
    for r in all_results:
        line = f"{r['level']:<12} {r['cells']:>10,}"
        for v in variants:
            vr = r["variants"].get(v["name"], {})
            res = vr.get("avg_final_residual", 0)
            line += f" | {res:>26.2e}" if res > 0 else f" | {'FAILED':>26}"
        print(line)

    # Speedup
    print(f"\n--- Speedup vs CPU GAMG ---")
    header = f"{'Level':<12} {'Cells':>10}"
    for v in variants:
        header += f" | {v['label']:>26}"
    print(header)
    for r in all_results:
        line = f"{r['level']:<12} {r['cells']:>10,}"
        for v in variants:
            vr = r["variants"].get(v["name"], {})
            if v["name"] == "cpu":
                line += f" | {'1.00x (ref)':>26}"
            else:
                sp = vr.get("speedup_vs_cpu", 0)
                line += f" | {sp:>25.2f}x" if sp > 0 else f" | {'N/A':>26}"
        print(line)

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Variable-coefficient spectral solver on buoyantCavity"
    )
    parser.add_argument("--docker", action="store_true", help="Run via Docker")
    parser.add_argument("--image", default="mixfoam:latest", help="Docker image")
    parser.add_argument("--timesteps", type=int, default=20,
                        help="Timesteps per run (default: 20)")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Timeout per run in seconds")
    parser.add_argument("--output-dir",
                        default=str(Path(__file__).parent / "buoyant_cavity_results"),
                        help="Output directory")
    parser.add_argument("--levels", nargs="+",
                        help="Specific levels (e.g., base 2x)")
    parser.add_argument("--variants", nargs="+",
                        help="Specific variants (e.g., cpu gpu_spectral)")
    parser.add_argument("--precision", default="FP32",
                        choices=["FP32", "FP64", "MIXED"])
    parser.add_argument("--n-correctors", type=int, default=0,
                        help="Override PIMPLE nCorrectors (0 = use default)")
    parser.add_argument("--lib-override",
                        help="Path to updated libOGL.so to mount into container")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_case = (
        Path(__file__).parent.parent / "tutorials" / "fluid" / "buoyantCavity"
    )

    bench_config = BenchmarkConfig(
        max_timesteps=args.timesteps,
        precision_policy=args.precision,
        iterative_refinement=args.precision != "FP64",
        debug_level=1 if args.verbose else 0,
    )

    extra_volumes = None
    if args.lib_override:
        lib_path = Path(args.lib_override).resolve()
        target = "/root/OpenFOAM/root-13/platforms/linux64GccDPInt32Opt/lib/libOGL.so"
        extra_volumes = [f"{lib_path}:{target}:ro"]

    run_config = RunConfig(
        use_docker=args.docker,
        docker_image=args.image,
        timeout=args.timeout,
        verbose=args.verbose,
        extra_volumes=extra_volumes,
    )

    levels = MESH_LEVELS
    if args.levels:
        levels = [l for l in MESH_LEVELS if l["name"] in args.levels]

    variants = VARIANTS
    if args.variants:
        variants = [v for v in VARIANTS if v["name"] in args.variants]

    all_results = []
    for level in levels:
        result = run_level(
            base_case, level, output_dir,
            bench_config, run_config, variants,
            n_correctors=args.n_correctors,
        )
        all_results.append(result)

    # Save combined results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print_summary(all_results, variants)


if __name__ == "__main__":
    main()
