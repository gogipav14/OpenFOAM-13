#!/usr/bin/env python3
"""
FFT preconditioner validation on lid-driven cavity (constant-coefficient Poisson).

The cavity case has a constant-coefficient pressure equation (nabla^2 p = rhs)
which is ideal for FFT preconditioning. The periodic Laplacian eigenvalues are
exact for this operator, so FFT should dramatically reduce iterations vs BJ.

Starts with 2D cases (nz=1, empty BC) as the simplest validation, then scales
to 3D by converting frontAndBack from empty to wall.

Usage:
    python cavity_fft_study.py --docker --image mixfoam:latest
    python cavity_fft_study.py --docker --levels 2D_10k --variants cpu gpu_fft
    python cavity_fft_study.py --docker --levels 2D_10k 2D_40k 3D_80k
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from case_modifier import prepare_case, BenchmarkConfig
from runner import run_case_docker, RunConfig
from log_parser import parse_solver_log, compute_metrics, find_solver_log


# Domain dimensions (metres, after convertToMeters 0.1)
DOMAIN_X = 0.1   # vertices 0→1
DOMAIN_Y = 0.1   # vertices 0→1
DOMAIN_Z_2D = 0.01  # original z: vertices 0→0.1
DOMAIN_Z_3D = 0.1   # extended z for 3D: vertices 0→1

MESH_LEVELS = [
    # 2D cases (nz=1, keep empty BC on frontAndBack)
    {"name": "2D_400",  "nx": 20,  "ny": 20,  "nz": 1, "mode": "2d"},
    {"name": "2D_10k",  "nx": 100, "ny": 100, "nz": 1, "mode": "2d"},
    {"name": "2D_40k",  "nx": 200, "ny": 200, "nz": 1, "mode": "2d"},
    {"name": "2D_160k", "nx": 400, "ny": 400, "nz": 1, "mode": "2d"},
    # 3D cases (extend z, change frontAndBack to wall)
    {"name": "3D_64k",  "nx": 40,  "ny": 40,  "nz": 40,  "mode": "3d"},
    {"name": "3D_250k", "nx": 63,  "ny": 63,  "nz": 63,  "mode": "3d"},
    {"name": "3D_512k", "nx": 80,  "ny": 80,  "nz": 80,  "mode": "3d"},
    {"name": "3D_2M",   "nx": 126, "ny": 126, "nz": 126, "mode": "3d"},
]

PRECOND_VARIANTS = [
    {
        "name": "cpu",
        "label": "CPU (GAMG)",
        "variant": "cpu",
        "preconditioner": None,
    },
    {
        "name": "gpu_block_jacobi",
        "label": "GPU Block Jacobi (bs=4)",
        "variant": "gpu",
        "preconditioner": "blockJacobi",
        "block_size": 4,
    },
    {
        "name": "gpu_fft",
        "label": "GPU FFT",
        "variant": "gpu",
        "preconditioner": "FFT",
        "needs_fft_dims": True,
    },
    {
        "name": "gpu_fft_bj",
        "label": "GPU FFT+BJ",
        "variant": "gpu",
        "preconditioner": "fftBlockJacobi",
        "block_size": 4,
        "needs_fft_dims": True,
    },
    {
        "name": "gpu_spectral",
        "label": "GPU Spectral (DCT direct)",
        "variant": "gpu_spectral",
        "needs_fft_dims": True,
    },
]


def switch_to_laminar(case_path: Path):
    """Switch turbulence model to laminar for constant-coefficient testing."""
    mt_file = case_path / "constant" / "momentumTransport"
    if mt_file.exists():
        mt_file.write_text(
            "FoamFile\n{\n    format      ascii;\n"
            "    class       dictionary;\n"
            '    object      momentumTransport;\n}\n\n'
            "simulationType  laminar;\n"
        )
    # Remove turbulence field files (not needed for laminar)
    for f in ["k", "epsilon", "omega", "nut", "nuTilda"]:
        fpath = case_path / "0" / f
        if fpath.exists():
            fpath.unlink()


def update_mesh_2d(case_path: Path, nx: int, ny: int):
    """Update 2D cavity mesh (keep nz=1, keep empty BCs)."""
    bmdict = case_path / "system" / "blockMeshDict"
    content = bmdict.read_text()

    # Change cell counts only (keep nz=1)
    content = re.sub(
        r'(hex\s*\([^)]+\)\s*\()\d+\s+\d+\s+\d+(\)\s*simpleGrading)',
        rf'\g<1>{nx} {ny} 1\g<2>',
        content
    )
    bmdict.write_text(content)
    return nx * ny


def convert_cavity_to_3d(case_path: Path, nx: int, ny: int, nz: int):
    """Convert the 2D cavity case to 3D by modifying BCs and mesh."""

    # 1. Update blockMeshDict: extend z-depth and change cell counts
    bmdict = case_path / "system" / "blockMeshDict"
    content = bmdict.read_text()

    # Change z-vertices from 0.1 to 1.0 (will be 0.1m after convertToMeters 0.1)
    content = content.replace("(0 0 0.1)", "(0 0 1)")
    content = content.replace("(1 0 0.1)", "(1 0 1)")
    content = content.replace("(1 1 0.1)", "(1 1 1)")
    content = content.replace("(0 1 0.1)", "(0 1 1)")

    # Change cell counts
    content = re.sub(
        r'(hex\s*\([^)]+\)\s*\()\d+\s+\d+\s+\d+(\)\s*simpleGrading)',
        rf'\g<1>{nx} {ny} {nz}\g<2>',
        content
    )

    # Change frontAndBack from empty to wall
    content = content.replace(
        "frontAndBack\n    {\n        type empty;",
        "frontAndBack\n    {\n        type wall;"
    )

    bmdict.write_text(content)

    # 2. Update all field files in 0/: change frontAndBack from empty to wall BCs
    zero_dir = case_path / "0"
    for field_file in zero_dir.iterdir():
        if not field_file.is_file():
            continue
        content = field_file.read_text()
        if "frontAndBack" not in content or "empty" not in content:
            continue

        field_name = field_file.name
        if field_name == "U":
            replacement = "frontAndBack\n    {\n        type            noSlip;\n    }"
        elif field_name == "p":
            replacement = "frontAndBack\n    {\n        type            zeroGradient;\n    }"
        elif field_name == "nut":
            replacement = (
                "frontAndBack\n    {\n"
                "        type            nutkWallFunction;\n"
                "        value           uniform 0;\n    }"
            )
        elif field_name in ("k", "epsilon", "omega"):
            wf_type = {
                "k": "kqRWallFunction",
                "epsilon": "epsilonWallFunction",
                "omega": "omegaWallFunction",
            }[field_name]
            replacement = (
                f"frontAndBack\n    {{\n"
                f"        type            {wf_type};\n"
                f"        value           $internalField;\n    }}"
            )
        else:
            replacement = "frontAndBack\n    {\n        type            zeroGradient;\n    }"

        content = content.replace(
            "frontAndBack\n    {\n        type            empty;\n    }",
            replacement
        )
        field_file.write_text(content)

    return nx * ny * nz


def run_precond_variant(
    base_case: Path,
    level: dict,
    precond_var: dict,
    output_dir: Path,
    bench_config: BenchmarkConfig,
    run_config: RunConfig,
    use_laminar: bool = False,
) -> dict:
    """Run a single preconditioner variant at a given mesh level."""
    level_name = level["name"]
    var_name = precond_var["name"]
    variant = precond_var["variant"]
    nx, ny, nz = level["nx"], level["ny"], level["nz"]
    mode = level.get("mode", "3d")

    var_dir = output_dir / f"{level_name}_{var_name}"

    # Compute FFT dimensions and mesh spacing
    fft_dims = None
    mesh_spacing = None
    if precond_var.get("needs_fft_dims"):
        fft_dims = (nx, ny, nz)
        domain_z = DOMAIN_Z_2D if mode == "2d" else DOMAIN_Z_3D
        mesh_spacing = (DOMAIN_X / nx, DOMAIN_Y / ny, domain_z / max(nz, 1))

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
        preconditioner=precond_var.get("preconditioner", "Jacobi") or "Jacobi",
        block_size=precond_var.get("block_size", 4),
        isai_sparsity_power=precond_var.get("isai_sparsity_power", 1),
        fft_dimensions=fft_dims,
        mesh_spacing=mesh_spacing,
    )

    print(f"    Preparing {var_name}...")
    prepare_case(base_case, var_dir, variant, config)

    if use_laminar:
        switch_to_laminar(var_dir)

    # Set mesh size and convert to 3D if needed
    if mode == "3d":
        actual_cells = convert_cavity_to_3d(var_dir, nx, ny, nz)
    else:
        actual_cells = update_mesh_2d(var_dir, nx, ny)

    # Force ascii write for log parsing and set writeInterval.
    # CFL conditioning is handled globally by case_modifier._enforce_cfl().
    controldict = var_dir / "system" / "controlDict"
    content = controldict.read_text()
    content = re.sub(r'writeFormat\s+\w+\s*;', 'writeFormat     ascii;', content)
    content = re.sub(
        r'writeInterval\s+\d+\s*;',
        f'writeInterval   {config.max_timesteps};',
        content,
    )
    controldict.write_text(content)

    print(f"    Running {var_name} ({actual_cells:,} cells, {mode})...")

    if run_config.use_docker:
        metrics = run_case_docker(var_dir, run_config, variant=variant)
    else:
        from runner import run_case_native
        metrics = run_case_native(var_dir, run_config)

    metrics.case_name = f"cavity_{level_name}"
    metrics.variant = var_name

    result = {
        "variant": var_name,
        "label": precond_var["label"],
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
              f"solver={metrics.solver_type}")
    else:
        print(f"    {var_name}: FAILED - {metrics.error_message}")

    # Save per-variant metrics
    metrics_path = var_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Save convergence data
    save_convergence_data(var_dir, output_dir / f"convergence_{level_name}_{var_name}.csv")

    return result


def save_convergence_data(case_dir: Path, output_path: Path):
    """Extract per-timestep convergence data from solver log."""
    log_file = find_solver_log(case_dir)
    if not log_file:
        return

    timesteps = parse_solver_log(log_file)
    if not timesteps:
        return

    lines = ["step,time,exec_time,p_initial_res,p_final_res,p_iters,cumulative_iters"]
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


def run_scaling_level(
    base_case: Path,
    level: dict,
    output_dir: Path,
    bench_config: BenchmarkConfig,
    run_config: RunConfig,
    precond_variants: list,
    use_laminar: bool = False,
) -> dict:
    """Run all preconditioner variants at a given mesh level."""
    level_name = level["name"]
    nx, ny, nz = level["nx"], level["ny"], level["nz"]
    expected_cells = nx * ny * nz
    mode = level.get("mode", "3d")

    print(f"\n{'='*70}")
    print(f"  Cavity {mode.upper()}: {level_name} "
          f"({nx}x{ny}x{nz} = {expected_cells:,} cells)")
    print(f"{'='*70}")

    result = {
        "level": level_name,
        "cells": expected_cells,
        "nx": nx, "ny": ny, "nz": nz,
        "mode": mode,
        "variants": {},
    }

    for pv in precond_variants:
        var_result = run_precond_variant(
            base_case, level, pv, output_dir, bench_config, run_config,
            use_laminar=use_laminar,
        )
        result["variants"][pv["name"]] = var_result

    # Compute speedups relative to CPU
    cpu_result = result["variants"].get("cpu")
    if cpu_result and cpu_result.get("completed"):
        cpu_t = cpu_result["time_per_step"]
        for name, vr in result["variants"].items():
            if name != "cpu" and vr.get("completed") and vr["time_per_step"] > 0:
                vr["speedup_vs_cpu"] = cpu_t / vr["time_per_step"]
            else:
                vr["speedup_vs_cpu"] = 0

    # Compute iteration reduction relative to BJ
    bj_result = result["variants"].get("gpu_block_jacobi")
    if bj_result and bj_result.get("completed"):
        bj_iters = bj_result["iters_per_step"]
        for name, vr in result["variants"].items():
            if vr.get("completed") and vr["iters_per_step"] > 0:
                vr["iter_ratio_vs_bj"] = bj_iters / vr["iters_per_step"]
            else:
                vr["iter_ratio_vs_bj"] = 0

    return result


def print_summary(all_results: list):
    """Print summary table."""
    print(f"\n{'='*90}")
    print("CAVITY FFT PRECONDITIONER VALIDATION (constant-coefficient Poisson)")
    print(f"{'='*90}")

    print(f"\n--- Wall-time per step [s] ---")
    header = f"{'Level':<12} {'Cells':>10}"
    for pv in PRECOND_VARIANTS:
        header += f" | {pv['label']:>22}"
    print(header)
    for r in all_results:
        line = f"{r['level']:<12} {r['cells']:>10,}"
        for pv in PRECOND_VARIANTS:
            vr = r["variants"].get(pv["name"], {})
            t = vr.get("time_per_step", 0)
            line += f" | {t:>22.4f}" if t > 0 else f" | {'FAILED':>22}"
        print(line)

    print(f"\n--- Pressure iterations per step ---")
    header = f"{'Level':<12} {'Cells':>10}"
    for pv in PRECOND_VARIANTS:
        header += f" | {pv['label']:>22}"
    print(header)
    for r in all_results:
        line = f"{r['level']:<12} {r['cells']:>10,}"
        for pv in PRECOND_VARIANTS:
            vr = r["variants"].get(pv["name"], {})
            it = vr.get("iters_per_step", 0)
            line += f" | {it:>22.1f}" if it > 0 else f" | {'FAILED':>22}"
        print(line)

    print(f"\n--- Iteration reduction vs Block Jacobi ---")
    header = f"{'Level':<12} {'Cells':>10}"
    for pv in PRECOND_VARIANTS:
        header += f" | {pv['label']:>22}"
    print(header)
    for r in all_results:
        line = f"{r['level']:<12} {r['cells']:>10,}"
        for pv in PRECOND_VARIANTS:
            vr = r["variants"].get(pv["name"], {})
            ir = vr.get("iter_ratio_vs_bj", 0)
            if pv["name"] == "gpu_block_jacobi":
                line += f" | {'1.00x (ref)':>22}"
            elif ir > 0:
                line += f" | {ir:>21.2f}x"
            else:
                line += f" | {'N/A':>22}"
        print(line)

    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(
        description="FFT preconditioner validation: cavity (constant-coefficient Poisson)"
    )
    parser.add_argument("--docker", action="store_true", help="Run via Docker")
    parser.add_argument("--image", default="mixfoam:latest", help="Docker image")
    parser.add_argument("--timesteps", type=int, default=20,
                        help="Timesteps per run (default: 20)")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Timeout per run in seconds (default: 1200)")
    parser.add_argument("--output-dir",
                        default=str(Path(__file__).parent / "cavity_fft_results"),
                        help="Output directory")
    parser.add_argument("--levels", nargs="+",
                        help="Specific levels (e.g., 2D_10k 3D_64k)")
    parser.add_argument("--variants", nargs="+",
                        help="Specific variants (e.g., cpu gpu_fft gpu_fft_bj)")
    parser.add_argument("--precision", default="FP32",
                        choices=["FP32", "FP64", "MIXED"])
    parser.add_argument("--laminar", action="store_true",
                        help="Use laminar flow (no turbulence model)")
    parser.add_argument("--lib-override",
                        help="Path to updated libOGL.so to mount into container")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_case = Path(__file__).parent.parent / "tutorials" / "incompressibleFluid" / "cavity"

    bench_config = BenchmarkConfig(
        max_timesteps=args.timesteps,
        precision_policy=args.precision,
        iterative_refinement=args.precision != "FP64",
        debug_level=1 if args.verbose else 0,
    )

    extra_volumes = None
    if args.lib_override:
        lib_path = Path(args.lib_override).resolve()
        # Mount into the FOAM_USER_LIBBIN location inside the container
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

    precond_variants = PRECOND_VARIANTS
    if args.variants:
        precond_variants = [v for v in PRECOND_VARIANTS if v["name"] in args.variants]

    all_results = []
    for level in levels:
        result = run_scaling_level(
            base_case, level, output_dir,
            bench_config, run_config, precond_variants,
            use_laminar=args.laminar,
        )
        all_results.append(result)

    # Save combined results
    results_path = output_dir / "preconditioner_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print_summary(all_results)

    print(f"\nTo generate plots, run:")
    print(f"  python plot_preconditioner.py --data-dir {output_dir}")


if __name__ == "__main__":
    main()
