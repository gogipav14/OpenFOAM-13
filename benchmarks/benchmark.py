#!/usr/bin/env python3
"""
OpenFOAM Tutorial Benchmark Suite

Compares CPU (PCG+DIC) vs GPU (OGLPCG via Ginkgo) performance across
OpenFOAM tutorials. Measures NFE (solver iterations) and wall-clock time.

Usage:
    # Scan tutorials and show which are GPU-compatible
    python benchmark.py scan

    # Run benchmarks on all compatible tutorials (Docker mode)
    python benchmark.py run --docker --image mixfoam:latest

    # Run benchmarks on a single tutorial
    python benchmark.py run --case fluid/cavity --docker

    # Run benchmarks on a category
    python benchmark.py run --category incompressibleFluid --docker

    # Run with native OpenFOAM (must be sourced)
    python benchmark.py run --case fluid/cavity --native

    # Quick test: run cavity only, 10 timesteps
    python benchmark.py run --case fluid/cavity --docker --timesteps 10

    # Generate report from previous results
    python benchmark.py report --results-dir ./benchmark_results

    # Dry run: show what would be done without running
    python benchmark.py run --dry-run
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the benchmarks directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tutorial_scanner import scan_tutorials, classify_cases, print_summary, TutorialCase
from case_modifier import prepare_case, BenchmarkConfig, create_allrun_benchmark
from log_parser import (
    parse_solver_log, compute_metrics, find_solver_log,
    BenchmarkMetrics, format_metrics_table,
)
from runner import run_case_docker, run_case_native, run_case_dev_container, RunConfig
from report_generator import (
    compare_results, ComparisonResult,
    write_csv, write_json,
    print_comparison_table, generate_nfe_vs_walltime_data,
)


def _run_case(case_dir: Path, run_config: RunConfig, variant: str = "cpu") -> BenchmarkMetrics:
    """Route to the appropriate runner based on config."""
    if run_config.use_dev_container:
        return run_case_dev_container(case_dir, run_config, variant=variant)
    elif run_config.use_docker:
        return run_case_docker(case_dir, run_config, variant=variant)
    else:
        return run_case_native(case_dir, run_config)


def _build_run_config(args) -> RunConfig:
    """Build RunConfig from parsed arguments, handling --dev flag."""
    use_dev = getattr(args, 'dev', False)
    return RunConfig(
        use_docker=getattr(args, 'docker', False) and not use_dev,
        docker_image=getattr(args, 'image', 'mixfoam:latest'),
        gpu_runtime=True,
        use_dev_container=use_dev,
        dev_container_name=getattr(args, 'dev_container', 'ogl-dev'),
        timeout=getattr(args, 'timeout', 600),
        dry_run=getattr(args, 'dry_run', False),
        verbose=getattr(args, 'verbose', False),
    )


def cmd_scan(args):
    """Scan tutorials and display classification."""
    tutorials_dir = Path(args.tutorials_dir)
    cases = scan_tutorials(tutorials_dir)
    groups = classify_cases(cases)
    print_summary(cases, groups)

    if args.list_all:
        print("\nAll GPU-ready cases:")
        print(f"{'Case':<60} {'Solver':<10} {'P-Field':<8} {'Cells':>8}")
        print("─" * 90)
        for case in sorted(groups["gpu_ready"], key=lambda c: c.name):
            print(
                f"{case.name:<60} {case.pressure_solver:<10} "
                f"{case.pressure_field:<8} {case.estimated_cells:>8}"
            )


def cmd_run(args):
    """Run benchmarks."""
    tutorials_dir = Path(args.tutorials_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Scan tutorials
    all_cases = scan_tutorials(tutorials_dir)
    groups = classify_cases(all_cases)

    # Filter cases based on arguments
    cases_to_run = _filter_cases(groups["gpu_ready"], args)

    if not cases_to_run:
        print("No matching cases found to benchmark.")
        return

    print(f"\nWill benchmark {len(cases_to_run)} cases")
    print(f"Results directory: {results_dir}")
    use_dev = getattr(args, 'dev', False)
    mode = 'Dev container' if use_dev else ('Docker' if args.docker else 'Native')
    print(f"Mode: {mode}")
    print(f"Timesteps per case: {args.timesteps}")
    print()

    # Configure
    bench_config = BenchmarkConfig(
        max_timesteps=args.timesteps,
        force_serial=not args.parallel,
        precision_policy=args.precision,
        iterative_refinement=args.precision != "FP64",
        debug_level=1 if args.verbose else 0,
    )

    run_config = _build_run_config(args)

    # Run benchmarks
    all_results = []
    total = len(cases_to_run)

    for i, case in enumerate(cases_to_run, 1):
        print(f"\n[{i}/{total}] {case.name}")
        print(f"  Original solver: {case.pressure_solver} | "
              f"Field: {case.pressure_field} | Cells: ~{case.estimated_cells}")

        try:
            result = _run_single_case(
                case, tutorials_dir, results_dir,
                bench_config, run_config, args
            )
            all_results.append(result)

            # Print intermediate result
            if result.cpu_ok and result.gpu_ok:
                print(f"  -> Speedup: {result.step_time_speedup:.2f}x "
                      f"(CPU: {result.cpu_time_per_step:.4f}s/step, "
                      f"GPU: {result.gpu_time_per_step:.4f}s/step)")
            elif result.cpu_ok:
                print(f"  -> CPU completed, GPU failed: {result.gpu_error}")
            elif result.gpu_ok:
                print(f"  -> GPU completed, CPU failed: {result.cpu_error}")
            else:
                print(f"  -> Both failed: CPU={result.cpu_error}, GPU={result.gpu_error}")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            all_results.append(ComparisonResult(
                case_name=case.name,
                category=case.category,
                cpu_error=str(e),
                gpu_error=str(e),
            ))

    # Generate reports
    print("\n\nGenerating reports...")
    _generate_reports(all_results, results_dir)


def _filter_cases(gpu_ready_cases: list, args) -> list:
    """Filter cases based on command-line arguments."""
    cases = gpu_ready_cases

    # Filter by specific case
    if args.case:
        cases = [c for c in cases if args.case in c.name]

    # Filter by category
    if args.category:
        cases = [c for c in cases if c.category == args.category]

    # Filter by max cell count
    if args.max_cells > 0:
        cases = [c for c in cases if c.estimated_cells <= args.max_cells or c.estimated_cells == 0]

    # Limit number of cases
    if args.limit > 0:
        cases = cases[:args.limit]

    return cases


def _run_single_case(
    case: TutorialCase,
    tutorials_dir: Path,
    results_dir: Path,
    bench_config: BenchmarkConfig,
    run_config: RunConfig,
    args,
) -> ComparisonResult:
    """Run CPU and GPU benchmarks for a single case."""

    case_results_dir = results_dir / case.name.replace("/", "_")
    case_results_dir.mkdir(parents=True, exist_ok=True)

    cpu_case_dir = case_results_dir / "cpu"
    gpu_case_dir = case_results_dir / "gpu"

    # Prepare CPU variant
    cpu_variant = getattr(args, 'cpu_variant', 'cpu_pcg')
    print(f"  Preparing CPU variant ({cpu_variant})...")
    prepare_case(case.path, cpu_case_dir, cpu_variant, bench_config)
    create_allrun_benchmark(cpu_case_dir, cpu_variant)

    # Prepare GPU variant
    print("  Preparing GPU variant...")
    prepare_case(case.path, gpu_case_dir, "gpu", bench_config)
    create_allrun_benchmark(gpu_case_dir, "gpu")

    # Run CPU
    print("  Running CPU baseline...")
    cpu_metrics = _run_case(cpu_case_dir, run_config, variant="cpu")
    cpu_metrics.case_name = case.name
    cpu_metrics.variant = "cpu"

    # Run GPU
    print("  Running GPU (OGLPCG)...")
    gpu_metrics = _run_case(gpu_case_dir, run_config, variant="gpu")
    gpu_metrics.case_name = case.name
    gpu_metrics.variant = "gpu"

    # Save individual metrics
    _save_metrics(cpu_metrics, case_results_dir / "cpu_metrics.json")
    _save_metrics(gpu_metrics, case_results_dir / "gpu_metrics.json")

    # Compare
    return compare_results(cpu_metrics, gpu_metrics, category=case.category)


def _save_metrics(metrics: BenchmarkMetrics, path: Path):
    """Save metrics to JSON (excluding raw timestep data)."""
    data = {
        'case_name': metrics.case_name,
        'variant': metrics.variant,
        'solver_type': metrics.solver_type,
        'total_wall_time': metrics.total_wall_time,
        'total_exec_time': metrics.total_exec_time,
        'mesh_time': metrics.mesh_time,
        'num_timesteps': metrics.num_timesteps,
        'avg_time_per_step': metrics.avg_time_per_step,
        'min_time_per_step': metrics.min_time_per_step,
        'max_time_per_step': metrics.max_time_per_step,
        'total_pressure_iters': metrics.total_pressure_iters,
        'avg_iters_per_step': metrics.avg_iters_per_step,
        'avg_iters_per_solve': metrics.avg_iters_per_solve,
        'total_pressure_solves': metrics.total_pressure_solves,
        'avg_initial_residual': metrics.avg_initial_residual,
        'avg_final_residual': metrics.avg_final_residual,
        'completed': metrics.completed,
        'error_message': metrics.error_message,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _generate_reports(results: list, results_dir: Path):
    """Generate all report files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV comparison
    csv_path = results_dir / f"comparison_{timestamp}.csv"
    write_csv(results, csv_path)

    # JSON results
    json_path = results_dir / f"comparison_{timestamp}.json"
    write_json(results, json_path)

    # NFE vs wall-time data for plotting
    nfe_path = results_dir / f"nfe_vs_walltime_{timestamp}.csv"
    generate_nfe_vs_walltime_data(results, nfe_path)

    # Terminal summary
    print_comparison_table(results)

    # Save a latest symlink
    latest_csv = results_dir / "comparison_latest.csv"
    latest_json = results_dir / "comparison_latest.json"
    latest_nfe = results_dir / "nfe_vs_walltime_latest.csv"
    for src, dst in [(csv_path, latest_csv), (json_path, latest_json), (nfe_path, latest_nfe)]:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.name)


def cmd_report(args):
    """Generate report from existing results."""
    results_dir = Path(args.results_dir)

    # Load all metrics from subdirectories
    results = []
    for case_dir in sorted(results_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        cpu_json = case_dir / "cpu_metrics.json"
        gpu_json = case_dir / "gpu_metrics.json"

        if not cpu_json.exists() and not gpu_json.exists():
            continue

        cpu_metrics = _load_metrics(cpu_json) if cpu_json.exists() else BenchmarkMetrics()
        gpu_metrics = _load_metrics(gpu_json) if gpu_json.exists() else BenchmarkMetrics()

        result = compare_results(cpu_metrics, gpu_metrics)
        results.append(result)

    if results:
        _generate_reports(results, results_dir)
    else:
        print("No results found in", results_dir)


def _load_metrics(path: Path) -> BenchmarkMetrics:
    """Load metrics from JSON."""
    with open(path) as f:
        data = json.load(f)

    metrics = BenchmarkMetrics()
    for key, value in data.items():
        if hasattr(metrics, key):
            setattr(metrics, key, value)
    return metrics


def _adjust_scaling_controls(case_dir: Path, scaled_dt: float, factor: int, max_timesteps: int):
    """
    Adjust controlDict and fvSolution for mesh scaling:
    - Enable adjustTimeStep with maxCo target
    - Scale deltaT proportional to mesh refinement
    - Increase maxIter for pressure solver
    """
    import re as re_mod

    # --- controlDict: enable adjustTimeStep + maxCo ---
    cdict_path = case_dir / "system" / "controlDict"
    if cdict_path.exists():
        content = cdict_path.read_text()

        # Replace deltaT with scaled value
        content = re_mod.sub(
            r'deltaT\s+[\d.eE+-]+\s*;',
            f'deltaT          {scaled_dt};',
            content,
        )

        # Set endTime based on scaled deltaT
        start_match = re_mod.search(r'startTime\s+([\d.eE+-]+)\s*;', content)
        start = float(start_match.group(1)) if start_match else 0
        new_end = start + max_timesteps * scaled_dt
        content = re_mod.sub(
            r'endTime\s+[\d.eE+-]+\s*;',
            f'endTime         {new_end};',
            content,
        )

        # Add adjustTimeStep and maxCo if not present
        if 'adjustTimeStep' not in content:
            # Insert before the closing of the controlDict (before last //)
            insert = (
                f'\nadjustTimeStep  yes;\n'
                f'maxCo           0.5;\n'
                f'maxDeltaT       {scaled_dt * 10};\n'
            )
            # Find the last // line
            last_comment = content.rfind('// ***')
            if last_comment > 0:
                content = content[:last_comment] + insert + '\n' + content[last_comment:]
            else:
                content += insert

        cdict_path.write_text(content)

    # --- fvSolution: increase maxIter for pressure ---
    fvsol_path = case_dir / "system" / "fvSolution"
    if fvsol_path.exists():
        content = fvsol_path.read_text()
        # Add maxIter to any solver block that doesn't have it
        # For PCG/DICPCG/OGLPCG, default maxIter is 1000; increase to 10000
        if 'maxIter' not in content:
            content = content.replace(
                'relTol          0.1;',
                'relTol          0.1;\n        maxIter         10000;',
                1,  # only first occurrence (the non-Final entry)
            )
        content = content.replace(
            'relTol          0;',
            'relTol          0;\n        maxIter         10000;',
        )
        fvsol_path.write_text(content)


def cmd_scaling(args):
    """Run mesh scaling study: one case at multiple refinement levels."""
    import csv as csv_mod
    case_path = Path(args.case_path)
    if not case_path.exists():
        # Try relative to tutorials dir
        tutorials_dir = Path(args.tutorials_dir) if hasattr(args, 'tutorials_dir') else (
            Path(__file__).parent.parent / "tutorials"
        )
        case_path = tutorials_dir / args.case_path
    if not case_path.exists():
        print(f"Error: case not found: {case_path}")
        return

    blockmesh = case_path / "system" / "blockMeshDict"
    if not blockmesh.exists():
        print(f"Error: {case_path} has no system/blockMeshDict")
        return

    factors = [int(f) for f in args.factors.split(",")]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    bench_config = BenchmarkConfig(
        max_timesteps=args.timesteps,
        force_serial=True,
        precision_policy=args.precision,
        iterative_refinement=args.precision != "FP64",
        debug_level=1 if args.verbose else 0,
        preconditioner=args.preconditioner,
        mg_smoother=args.mg_smoother,
        mg_smoother_iters=args.mg_smoother_iters,
        mg_cache_interval=args.mg_cache_interval,
        mg_cache_max_iters=args.mg_cache_max_iters,
    )

    run_config = _build_run_config(args)

    cpu_variant = args.cpu_variant

    # Parse base blockMeshDict to find hex refinement counts
    bm_content = blockmesh.read_text()
    import re as re_mod
    hex_matches = re_mod.findall(
        r'(hex\s*\([^)]+\)\s*\()(\d+)\s+(\d+)\s+(\d+)(\)\s*)',
        bm_content,
    )
    if not hex_matches:
        print("Error: cannot parse hex blocks in blockMeshDict")
        return

    base_cells = sum(int(m[1]) * int(m[2]) * int(m[3]) for m in hex_matches)
    print(f"Base mesh: {base_cells} cells ({len(hex_matches)} hex blocks)")
    print(f"Refinement factors: {factors}")
    print(f"CPU variant: {cpu_variant}, GPU precision: {args.precision}")
    print()

    # Parse base controlDict for deltaT
    controldict_path = case_path / "system" / "controlDict"
    cdict_content = controldict_path.read_text() if controldict_path.exists() else ""
    dt_match = re_mod.search(r'deltaT\s+([\d.eE+-]+)\s*;', cdict_content)
    base_dt = float(dt_match.group(1)) if dt_match else 0.005

    all_results = []

    for factor in factors:
        # For 2D cases (nz=1), only refine in 2D
        is_2d = any(int(m[3]) == 1 for m in hex_matches)
        if is_2d:
            total_cells = base_cells * (factor ** 2)
        else:
            total_cells = base_cells * (factor ** 3)

        print(f"\n{'='*70}")
        print(f"Mesh factor {factor}x — ~{total_cells} cells")
        print(f"{'='*70}")

        # Create scaled blockMeshDict
        scaled_bm = bm_content
        for full_match_str, nx, ny, nz in [(m[0]+m[1]+" "+m[2]+" "+m[3]+m[4], m[1], m[2], m[3]) for m in hex_matches]:
            new_nx = int(nx) * factor
            new_ny = int(ny) * factor
            new_nz = int(nz) * factor if int(nz) > 1 else int(nz)
            old_counts = f"{nx} {ny} {nz}"
            new_counts = f"{new_nx} {new_ny} {new_nz}"
            scaled_bm = scaled_bm.replace(old_counts, new_counts, 1)

        # Scale deltaT by 1/factor to maintain constant Courant number
        scaled_dt = base_dt / factor

        # Prepare CPU and GPU case directories
        case_label = case_path.name
        factor_dir = results_dir / f"{case_label}_x{factor}"
        cpu_dir = factor_dir / "cpu"
        gpu_dir = factor_dir / "gpu"

        # CPU variant
        print(f"  Preparing CPU ({cpu_variant})...")
        prepare_case(case_path, cpu_dir, cpu_variant, bench_config)
        create_allrun_benchmark(cpu_dir, cpu_variant)
        (cpu_dir / "system" / "blockMeshDict").write_text(scaled_bm)
        _adjust_scaling_controls(cpu_dir, scaled_dt, factor, bench_config.max_timesteps)

        # GPU variant
        print(f"  Preparing GPU (OGLPCG, {args.precision})...")
        prepare_case(case_path, gpu_dir, "gpu", bench_config)
        create_allrun_benchmark(gpu_dir, "gpu")
        (gpu_dir / "system" / "blockMeshDict").write_text(scaled_bm)
        _adjust_scaling_controls(gpu_dir, scaled_dt, factor, bench_config.max_timesteps)

        # Run CPU
        print(f"  Running CPU...")
        cpu_metrics = _run_case(cpu_dir, run_config, variant="cpu")
        cpu_metrics.case_name = f"{case_label}_x{factor}"
        cpu_metrics.variant = "cpu"

        # Run GPU
        print(f"  Running GPU...")
        gpu_metrics = _run_case(gpu_dir, run_config, variant="gpu")
        gpu_metrics.case_name = f"{case_label}_x{factor}"
        gpu_metrics.variant = "gpu"

        # Save metrics
        _save_metrics(cpu_metrics, factor_dir / "cpu_metrics.json")
        _save_metrics(gpu_metrics, factor_dir / "gpu_metrics.json")

        result = compare_results(cpu_metrics, gpu_metrics, category=case_label)
        all_results.append((factor, total_cells, result))

        # Print result
        if result.cpu_ok and result.gpu_ok:
            print(f"  CPU: {result.cpu_time_per_step:.4f} s/step, "
                  f"{result.cpu_iters_per_step:.0f} it/step, "
                  f"res={result.cpu_avg_residual:.2e}")
            print(f"  GPU: {result.gpu_time_per_step:.4f} s/step, "
                  f"{result.gpu_iters_per_step:.0f} it/step, "
                  f"res={result.gpu_avg_residual:.2e}")
            print(f"  Step speedup: {result.step_time_speedup:.2f}x | "
                  f"NFE cost ratio: {result.nfe_cost_ratio:.3f} | "
                  f"Res ratio: {result.residual_ratio:.2e}")
        elif result.cpu_ok:
            print(f"  CPU OK, GPU failed: {result.gpu_error}")
        elif result.gpu_ok:
            print(f"  GPU OK, CPU failed: {result.cpu_error}")
        else:
            print(f"  Both failed: CPU={result.cpu_error}, GPU={result.gpu_error}")

    # Write scaling CSV
    print(f"\n{'='*70}")
    print("MESH SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Factor':>6} {'Cells':>10} {'CPU s/step':>10} {'GPU s/step':>10} "
          f"{'Speedup':>8} {'CPU it/s':>8} {'GPU it/s':>8} "
          f"{'NFE cost':>8} {'Res ratio':>10}")
    print("─" * 90)

    csv_path = results_dir / "scaling_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv_mod.writer(f)
        writer.writerow([
            'factor', 'cells',
            'cpu_time_per_step', 'gpu_time_per_step', 'step_speedup',
            'cpu_iters_per_step', 'gpu_iters_per_step',
            'cpu_time_per_solve', 'gpu_time_per_solve',
            'cpu_time_per_nfe', 'gpu_time_per_nfe', 'nfe_cost_ratio',
            'cpu_avg_residual', 'gpu_avg_residual', 'residual_ratio',
        ])
        for factor, cells, r in all_results:
            if r.cpu_ok and r.gpu_ok:
                nfe_str = f"{r.nfe_cost_ratio:.3f}"
                print(
                    f"{factor:>6} {cells:>10} {r.cpu_time_per_step:>10.4f} "
                    f"{r.gpu_time_per_step:>10.4f} {r.step_time_speedup:>7.2f}x "
                    f"{r.cpu_iters_per_step:>8.0f} {r.gpu_iters_per_step:>8.0f} "
                    f"{nfe_str:>8} {r.residual_ratio:>10.2e}"
                )
                writer.writerow([
                    factor, cells,
                    r.cpu_time_per_step, r.gpu_time_per_step, r.step_time_speedup,
                    r.cpu_iters_per_step, r.gpu_iters_per_step,
                    r.cpu_time_per_solve, r.gpu_time_per_solve,
                    r.cpu_time_per_nfe, r.gpu_time_per_nfe, r.nfe_cost_ratio,
                    r.cpu_avg_residual, r.gpu_avg_residual, r.residual_ratio,
                ])
            else:
                status = "FAIL"
                if not r.cpu_ok:
                    status += f" CPU:{r.cpu_error}"
                if not r.gpu_ok:
                    status += f" GPU:{r.gpu_error}"
                print(f"{factor:>6} {cells:>10} {status}")

    print(f"\nScaling results written to: {csv_path}")


def cmd_compare(args):
    """Run a single case with CPU and GPU variants side-by-side."""
    case_path = Path(args.case_path).resolve()
    if not case_path.exists():
        print(f"Error: case not found: {case_path}")
        return

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    bench_config = BenchmarkConfig(
        max_timesteps=args.timesteps,
        force_serial=True,
        precision_policy=args.precision,
        iterative_refinement=args.precision != "FP64",
        debug_level=1 if args.verbose else 0,
        preconditioner=args.preconditioner,
        mg_smoother=args.mg_smoother,
        mg_smoother_iters=args.mg_smoother_iters,
        mg_cache_interval=args.mg_cache_interval,
        mg_cache_max_iters=args.mg_cache_max_iters,
    )

    run_config = _build_run_config(args)

    cpu_variant = args.cpu_variant
    case_label = case_path.name

    cpu_dir = results_dir / "cpu"
    gpu_dir = results_dir / "gpu"

    # Prepare CPU variant
    print(f"Preparing CPU ({cpu_variant})...")
    prepare_case(case_path, cpu_dir, cpu_variant, bench_config)
    create_allrun_benchmark(cpu_dir, cpu_variant)

    # Prepare GPU variant
    print(f"Preparing GPU (OGLPCG, {args.precision})...")
    prepare_case(case_path, gpu_dir, "gpu", bench_config)
    create_allrun_benchmark(gpu_dir, "gpu")

    # Run CPU
    print(f"Running CPU...")
    cpu_metrics = _run_case(cpu_dir, run_config, variant="cpu")
    cpu_metrics.case_name = case_label
    cpu_metrics.variant = "cpu"

    # Run GPU
    print(f"Running GPU...")
    gpu_metrics = _run_case(gpu_dir, run_config, variant="gpu")
    gpu_metrics.case_name = case_label
    gpu_metrics.variant = "gpu"

    # Save metrics
    _save_metrics(cpu_metrics, results_dir / "cpu_metrics.json")
    _save_metrics(gpu_metrics, results_dir / "gpu_metrics.json")

    result = compare_results(cpu_metrics, gpu_metrics, category=case_label)

    # Print results
    print(f"\n{'='*70}")
    print(f"COMPARISON: {case_label}")
    print(f"{'='*70}")
    if result.cpu_ok and result.gpu_ok:
        print(f"  CPU ({cpu_variant}): {result.cpu_time_per_step:.4f} s/step, "
              f"{result.cpu_iters_per_step:.0f} it/step, "
              f"res={result.cpu_avg_residual:.2e}")
        print(f"  GPU (OGLPCG):      {result.gpu_time_per_step:.4f} s/step, "
              f"{result.gpu_iters_per_step:.0f} it/step, "
              f"res={result.gpu_avg_residual:.2e}")
        print(f"\n  Step speedup:   {result.step_time_speedup:.2f}x")
        print(f"  NFE cost ratio: {result.nfe_cost_ratio:.3f}")
        print(f"  Residual ratio: {result.residual_ratio:.2e}")
        if result.cpu_time_per_nfe > 0 and result.gpu_time_per_nfe > 0:
            print(f"  CPU ms/NFE:     {result.cpu_time_per_nfe*1000:.3f}")
            print(f"  GPU ms/NFE:     {result.gpu_time_per_nfe*1000:.3f}")
    else:
        if not result.cpu_ok:
            print(f"  CPU FAILED: {result.cpu_error}")
        if not result.gpu_ok:
            print(f"  GPU FAILED: {result.gpu_error}")

    print(f"\nResults saved to: {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenFOAM Tutorial Benchmark Suite: CPU vs GPU comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global arguments
    parser.add_argument(
        "--tutorials-dir",
        default=str(Path(__file__).parent.parent / "tutorials"),
        help="Path to OpenFOAM tutorials directory",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan and classify tutorials")
    scan_parser.add_argument("--list-all", action="store_true",
                             help="List all GPU-ready cases")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("--case", help="Run specific case (substring match)")
    run_parser.add_argument("--category", help="Run cases from specific category")
    run_parser.add_argument("--docker", action="store_true",
                            help="Run in Docker container")
    run_parser.add_argument("--dev", action="store_true",
                            help="Run in persistent dev container (./dev.sh start first)")
    run_parser.add_argument("--dev-container", default="ogl-dev",
                            help="Dev container name (default: ogl-dev)")
    run_parser.add_argument("--native", action="store_true",
                            help="Run natively (OpenFOAM must be sourced)")
    run_parser.add_argument("--image", default="mixfoam:latest",
                            help="Docker image name (default: mixfoam:latest)")
    run_parser.add_argument("--timesteps", type=int, default=50,
                            help="Number of timesteps per case (default: 50)")
    run_parser.add_argument("--timeout", type=int, default=600,
                            help="Timeout per case in seconds (default: 600)")
    run_parser.add_argument("--precision", default="FP64",
                            choices=["FP32", "FP64", "MIXED"],
                            help="GPU precision policy (default: FP64)")
    run_parser.add_argument("--cpu-variant", default="cpu_pcg",
                            choices=["cpu", "cpu_pcg"],
                            help="CPU baseline: 'cpu' keeps original GAMG/PCG, "
                                 "'cpu_pcg' forces PCG+DIC for algorithm-matched "
                                 "comparison (default: cpu_pcg)")
    run_parser.add_argument("--results-dir", default="./benchmark_results",
                            help="Results output directory")
    run_parser.add_argument("--max-cells", type=int, default=0,
                            help="Skip cases with more cells than this (0=no limit)")
    run_parser.add_argument("--limit", type=int, default=0,
                            help="Max number of cases to run (0=all)")
    run_parser.add_argument("--parallel", action="store_true",
                            help="Allow parallel (decomposePar) runs")
    run_parser.add_argument("--dry-run", action="store_true",
                            help="Show what would be done without running")
    run_parser.add_argument("--verbose", "-v", action="store_true",
                            help="Verbose output")

    # Report command
    report_parser = subparsers.add_parser("report",
                                          help="Generate report from existing results")
    report_parser.add_argument("--results-dir", default="./benchmark_results",
                               help="Results directory to read from")

    # Scaling command
    scale_parser = subparsers.add_parser(
        "scaling",
        help="Mesh scaling study: run one case at multiple mesh sizes",
    )
    scale_parser.add_argument("case_path",
                              help="Path to a tutorial case with blockMeshDict")
    scale_parser.add_argument("--factors", default="1,2,4,8",
                              help="Comma-separated mesh refinement factors "
                                   "(default: 1,2,4,8)")
    scale_parser.add_argument("--docker", action="store_true",
                              help="Run in Docker container")
    scale_parser.add_argument("--dev", action="store_true",
                              help="Run in persistent dev container (./dev.sh start first)")
    scale_parser.add_argument("--dev-container", default="ogl-dev",
                              help="Dev container name (default: ogl-dev)")
    scale_parser.add_argument("--native", action="store_true",
                              help="Run natively")
    scale_parser.add_argument("--image", default="mixfoam:latest",
                              help="Docker image name")
    scale_parser.add_argument("--timesteps", type=int, default=20,
                              help="Timesteps per mesh size (default: 20)")
    scale_parser.add_argument("--timeout", type=int, default=1200,
                              help="Timeout per case (default: 1200)")
    scale_parser.add_argument("--precision", default="FP64",
                              choices=["FP32", "FP64", "MIXED"],
                              help="GPU precision (default: FP64)")
    scale_parser.add_argument("--cpu-variant", default="cpu_pcg",
                              choices=["cpu", "cpu_pcg"],
                              help="CPU baseline variant (default: cpu_pcg)")
    scale_parser.add_argument("--preconditioner", default="blockJacobi",
                              help="GPU preconditioner (default: blockJacobi)")
    scale_parser.add_argument("--mg-smoother", default="jacobi",
                              choices=["jacobi", "chebyshev", "blockJacobi"],
                              help="Multigrid smoother type (default: jacobi)")
    scale_parser.add_argument("--mg-smoother-iters", type=int, default=2,
                              help="Multigrid smoother iterations/degree (default: 2)")
    scale_parser.add_argument("--mg-cache-interval", type=int, default=0,
                              help="MG hierarchy cache interval (0=rebuild every call)")
    scale_parser.add_argument("--mg-cache-max-iters", type=int, default=200,
                              help="Force MG rebuild if iters exceed this (default: 200)")
    scale_parser.add_argument("--results-dir", default="./scaling_results",
                              help="Results output directory")
    scale_parser.add_argument("--verbose", "-v", action="store_true",
                              help="Verbose output")

    # Compare command — single case CPU vs GPU
    compare_parser = subparsers.add_parser(
        "compare",
        help="Run a single case with CPU and GPU side-by-side",
    )
    compare_parser.add_argument("case_path",
                                help="Path to an OpenFOAM case directory")
    compare_parser.add_argument("--docker", action="store_true",
                                help="Run in Docker container")
    compare_parser.add_argument("--dev", action="store_true",
                                help="Run in persistent dev container (./dev.sh start first)")
    compare_parser.add_argument("--dev-container", default="ogl-dev",
                                help="Dev container name (default: ogl-dev)")
    compare_parser.add_argument("--native", action="store_true",
                                help="Run natively")
    compare_parser.add_argument("--image", default="mixfoam:latest",
                                help="Docker image name")
    compare_parser.add_argument("--timesteps", type=int, default=50,
                                help="Timesteps to run (default: 50)")
    compare_parser.add_argument("--timeout", type=int, default=3600,
                                help="Timeout per variant in seconds (default: 3600)")
    compare_parser.add_argument("--precision", default="FP64",
                                choices=["FP32", "FP64", "MIXED"],
                                help="GPU precision (default: FP64)")
    compare_parser.add_argument("--cpu-variant", default="cpu_pcg",
                                choices=["cpu", "cpu_pcg"],
                                help="CPU baseline variant (default: cpu_pcg)")
    compare_parser.add_argument("--preconditioner", default="blockJacobi",
                                help="GPU preconditioner (default: blockJacobi)")
    compare_parser.add_argument("--mg-smoother", default="jacobi",
                                choices=["jacobi", "chebyshev", "blockJacobi"],
                                help="Multigrid smoother type (default: jacobi)")
    compare_parser.add_argument("--mg-smoother-iters", type=int, default=2,
                                help="Multigrid smoother iterations/degree (default: 2)")
    compare_parser.add_argument("--mg-cache-interval", type=int, default=0,
                                help="MG hierarchy cache interval (0=rebuild every call)")
    compare_parser.add_argument("--mg-cache-max-iters", type=int, default=200,
                                help="Force MG rebuild if iters exceed this (default: 200)")
    compare_parser.add_argument("--results-dir", default="./compare_results",
                                help="Results output directory")
    compare_parser.add_argument("--verbose", "-v", action="store_true",
                                help="Verbose output")

    args = parser.parse_args()

    def _check_exec_mode(args):
        dev = getattr(args, 'dev', False)
        docker = getattr(args, 'docker', False)
        native = getattr(args, 'native', False)
        if not dev and not docker and not native:
            print("Error: specify --docker, --dev, or --native execution mode")
            sys.exit(1)

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "run":
        _check_exec_mode(args)
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "scaling":
        _check_exec_mode(args)
        cmd_scaling(args)
    elif args.command == "compare":
        _check_exec_mode(args)
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
