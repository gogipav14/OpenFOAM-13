#!/usr/bin/env python3
"""
Generate benchmark comparison reports:
- CSV output for data analysis
- Terminal summary table
- Speedup calculations (CPU baseline vs GPU)
"""

import csv
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from log_parser import BenchmarkMetrics


@dataclass
class ComparisonResult:
    """Comparison between CPU and GPU runs for a single case."""
    case_name: str
    category: str = ""

    # CPU metrics
    cpu_wall_time: float = 0.0
    cpu_time_per_step: float = 0.0
    cpu_total_iters: int = 0
    cpu_iters_per_step: float = 0.0
    cpu_solver: str = ""
    cpu_avg_residual: float = 0.0

    # GPU metrics
    gpu_wall_time: float = 0.0
    gpu_time_per_step: float = 0.0
    gpu_total_iters: int = 0
    gpu_iters_per_step: float = 0.0
    gpu_solver: str = ""
    gpu_avg_residual: float = 0.0

    # Per-unit metrics
    cpu_time_per_solve: float = 0.0    # wall_time / total_pressure_solves
    gpu_time_per_solve: float = 0.0
    cpu_time_per_nfe: float = 0.0      # wall_time / total_pressure_iters
    gpu_time_per_nfe: float = 0.0
    solve_time_speedup: float = 0.0    # cpu_time_per_solve / gpu_time_per_solve
    nfe_cost_ratio: float = 0.0        # gpu_time_per_nfe / cpu_time_per_nfe (< 1 = GPU cheaper per iter)

    # Comparison
    wall_time_speedup: float = 0.0  # cpu_wall / gpu_wall
    step_time_speedup: float = 0.0  # cpu_step / gpu_step
    iter_ratio: float = 0.0  # gpu_iters / cpu_iters (< 1 means GPU needs fewer)
    residual_ratio: float = 0.0  # gpu_res / cpu_res (~1 means same accuracy)

    # Status
    cpu_ok: bool = False
    gpu_ok: bool = False
    cpu_error: str = ""
    gpu_error: str = ""


def compare_results(
    cpu_metrics: BenchmarkMetrics,
    gpu_metrics: BenchmarkMetrics,
    category: str = "",
) -> ComparisonResult:
    """Compare CPU and GPU benchmark results."""
    result = ComparisonResult(
        case_name=cpu_metrics.case_name or gpu_metrics.case_name,
        category=category,
    )

    # CPU
    if cpu_metrics.completed:
        result.cpu_ok = True
        result.cpu_wall_time = cpu_metrics.total_wall_time
        result.cpu_time_per_step = cpu_metrics.avg_time_per_step
        result.cpu_total_iters = cpu_metrics.total_pressure_iters
        result.cpu_iters_per_step = cpu_metrics.avg_iters_per_step
        result.cpu_solver = cpu_metrics.solver_type
        result.cpu_avg_residual = cpu_metrics.avg_final_residual
    else:
        result.cpu_error = cpu_metrics.error_message

    # GPU
    if gpu_metrics.completed:
        result.gpu_ok = True
        result.gpu_wall_time = gpu_metrics.total_wall_time
        result.gpu_time_per_step = gpu_metrics.avg_time_per_step
        result.gpu_total_iters = gpu_metrics.total_pressure_iters
        result.gpu_iters_per_step = gpu_metrics.avg_iters_per_step
        result.gpu_solver = gpu_metrics.solver_type
        result.gpu_avg_residual = gpu_metrics.avg_final_residual
    else:
        result.gpu_error = gpu_metrics.error_message

    # Per-unit metrics (per-solve, per-NFE)
    if result.cpu_ok:
        if cpu_metrics.total_pressure_solves > 0:
            result.cpu_time_per_solve = (
                cpu_metrics.total_exec_time / cpu_metrics.total_pressure_solves
            )
        if cpu_metrics.total_pressure_iters > 0:
            result.cpu_time_per_nfe = (
                cpu_metrics.total_exec_time / cpu_metrics.total_pressure_iters
            )

    if result.gpu_ok:
        if gpu_metrics.total_pressure_solves > 0:
            result.gpu_time_per_solve = (
                gpu_metrics.total_exec_time / gpu_metrics.total_pressure_solves
            )
        if gpu_metrics.total_pressure_iters > 0:
            result.gpu_time_per_nfe = (
                gpu_metrics.total_exec_time / gpu_metrics.total_pressure_iters
            )

    # Compute speedups
    if result.cpu_ok and result.gpu_ok:
        if result.gpu_wall_time > 0:
            result.wall_time_speedup = result.cpu_wall_time / result.gpu_wall_time
        if result.gpu_time_per_step > 0:
            result.step_time_speedup = result.cpu_time_per_step / result.gpu_time_per_step
        if result.cpu_total_iters > 0:
            result.iter_ratio = result.gpu_total_iters / result.cpu_total_iters
        if result.cpu_avg_residual > 0:
            result.residual_ratio = result.gpu_avg_residual / result.cpu_avg_residual
        if result.gpu_time_per_solve > 0:
            result.solve_time_speedup = (
                result.cpu_time_per_solve / result.gpu_time_per_solve
            )
        if result.gpu_time_per_nfe > 0 and result.cpu_time_per_nfe > 0:
            result.nfe_cost_ratio = (
                result.gpu_time_per_nfe / result.cpu_time_per_nfe
            )

    return result


def write_csv(results: list, output_path: Path):
    """Write comparison results to CSV."""
    if not results:
        return

    fieldnames = [
        'case_name', 'category',
        'cpu_solver', 'gpu_solver',
        'cpu_wall_time', 'gpu_wall_time', 'wall_time_speedup',
        'cpu_time_per_step', 'gpu_time_per_step', 'step_time_speedup',
        'cpu_total_iters', 'gpu_total_iters', 'iter_ratio',
        'cpu_iters_per_step', 'gpu_iters_per_step',
        'cpu_time_per_solve', 'gpu_time_per_solve', 'solve_time_speedup',
        'cpu_time_per_nfe', 'gpu_time_per_nfe', 'nfe_cost_ratio',
        'cpu_avg_residual', 'gpu_avg_residual', 'residual_ratio',
        'cpu_ok', 'gpu_ok', 'cpu_error', 'gpu_error',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fieldnames}
            writer.writerow(row)

    print(f"CSV report written to: {output_path}")


def write_json(results: list, output_path: Path):
    """Write comparison results to JSON."""
    data = [asdict(r) for r in results]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"JSON report written to: {output_path}")


def print_comparison_table(results: list):
    """Print a formatted comparison table to terminal."""
    if not results:
        print("No results to display.")
        return

    # Filter to only completed comparisons
    valid = [r for r in results if r.cpu_ok and r.gpu_ok]
    failed = [r for r in results if not (r.cpu_ok and r.gpu_ok)]

    print(f"\n{'='*140}")
    print(f"BENCHMARK COMPARISON: CPU (PCG+DIC) vs GPU (OGLPCG) — Same Algorithm, Same Tolerance")
    print(f"{'='*140}")

    if valid:
        header = (
            f"{'Case':<32} {'CPU s/step':>10} {'GPU s/step':>10} "
            f"{'Speedup':>8} {'CPU it/s':>8} {'GPU it/s':>8} "
            f"{'ms/solve':>8} {'ms/NFE':>8} "
            f"{'NFE $':>7} {'Res ratio':>10}"
        )
        print(header)
        print("─" * 140)

        for r in sorted(valid, key=lambda x: x.step_time_speedup, reverse=True):
            name = r.case_name[:32] if len(r.case_name) > 32 else r.case_name
            speedup_str = f"{r.step_time_speedup:.2f}x"
            # Per-solve cost in ms
            gpu_ms_solve = r.gpu_time_per_solve * 1000 if r.gpu_time_per_solve > 0 else 0
            # Per-NFE cost in ms
            gpu_ms_nfe = r.gpu_time_per_nfe * 1000 if r.gpu_time_per_nfe > 0 else 0
            # NFE cost ratio (< 1 = GPU cheaper per iteration)
            nfe_str = f"{r.nfe_cost_ratio:.2f}" if r.nfe_cost_ratio > 0 else "N/A"
            print(
                f"{name:<32} {r.cpu_time_per_step:>10.4f} {r.gpu_time_per_step:>10.4f} "
                f"{speedup_str:>8} {r.cpu_iters_per_step:>8.1f} {r.gpu_iters_per_step:>8.1f} "
                f"{gpu_ms_solve:>8.2f} {gpu_ms_nfe:>8.4f} "
                f"{nfe_str:>7} {r.residual_ratio:>10.2e}"
            )

        # Summary statistics
        speedups = [r.step_time_speedup for r in valid if r.step_time_speedup > 0]
        nfe_costs = [r.nfe_cost_ratio for r in valid if r.nfe_cost_ratio > 0]
        if speedups:
            print("─" * 140)
            avg_spd = sum(speedups) / len(speedups)
            print(
                f"{'SUMMARY':<32} {'':>10} {'':>10} "
                f"{'avg=' + f'{avg_spd:.2f}x':>8} "
            )
            print(
                f"{'':>32} {'':>10} {'':>10} "
                f"{'min=' + f'{min(speedups):.2f}x':>8} "
            )
            print(
                f"{'':>32} {'':>10} {'':>10} "
                f"{'max=' + f'{max(speedups):.2f}x':>8} "
            )
        if nfe_costs:
            avg_nfe = sum(nfe_costs) / len(nfe_costs)
            print(
                f"{'NFE cost ratio (GPU/CPU)':<32} {'':>10} {'':>10} "
                f"{'':>8} {'':>8} {'':>8} {'':>8} {'':>8} "
                f"{'avg=' + f'{avg_nfe:.2f}':>7}"
            )

    if failed:
        print(f"\n{'─'*110}")
        print(f"FAILED CASES ({len(failed)}):")
        for r in failed:
            status = []
            if not r.cpu_ok:
                status.append(f"CPU: {r.cpu_error}")
            if not r.gpu_ok:
                status.append(f"GPU: {r.gpu_error}")
            print(f"  {r.case_name}: {'; '.join(status)}")

    print(f"\n{'='*110}")
    print(f"Total: {len(results)} cases, {len(valid)} compared, {len(failed)} failed")
    print(f"{'='*110}\n")


def generate_nfe_vs_walltime_data(results: list, output_path: Path):
    """
    Generate NFE vs wall-time data for plotting.
    Outputs a CSV with columns suitable for scatter/line plots.
    """
    fieldnames = [
        'case_name', 'category', 'variant',
        'total_nfe', 'total_wall_time',
        'nfe_per_step', 'wall_time_per_step',
        'num_timesteps',
    ]

    rows = []
    for r in results:
        if r.cpu_ok:
            rows.append({
                'case_name': r.case_name,
                'category': r.category,
                'variant': 'CPU',
                'total_nfe': r.cpu_total_iters,
                'total_wall_time': r.cpu_wall_time,
                'nfe_per_step': r.cpu_iters_per_step,
                'wall_time_per_step': r.cpu_time_per_step,
                'num_timesteps': int(r.cpu_wall_time / r.cpu_time_per_step)
                    if r.cpu_time_per_step > 0 else 0,
            })
        if r.gpu_ok:
            rows.append({
                'case_name': r.case_name,
                'category': r.category,
                'variant': 'GPU',
                'total_nfe': r.gpu_total_iters,
                'total_wall_time': r.gpu_wall_time,
                'nfe_per_step': r.gpu_iters_per_step,
                'wall_time_per_step': r.gpu_time_per_step,
                'num_timesteps': int(r.gpu_wall_time / r.gpu_time_per_step)
                    if r.gpu_time_per_step > 0 else 0,
            })

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"NFE vs wall-time data written to: {output_path}")
