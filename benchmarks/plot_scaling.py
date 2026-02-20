#!/usr/bin/env python3
"""
Plot mesh scaling benchmark results.

Generates 4 publication-quality figures:
1. Speedup vs mesh size (log-log)
2. Wall time per step vs mesh size (log-log, CPU vs GPU)
3. NFE cost ratio vs mesh size
4. Per-iteration cost vs mesh size

Usage:
    python plot_scaling.py --csv scaling_results_cavity/scaling_results.csv
    python plot_scaling.py --csv scaling_results_cavity/scaling_results.csv --output-dir plots/
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

CPU_COLOR = '#2166ac'
GPU_COLOR = '#b2182b'
RATIO_COLOR = '#4daf4a'


def load_scaling_csv(csv_path: Path) -> dict:
    """Load scaling results CSV into numpy arrays."""
    data = {
        'factor': [], 'cells': [],
        'cpu_time_per_step': [], 'gpu_time_per_step': [], 'step_speedup': [],
        'cpu_iters_per_step': [], 'gpu_iters_per_step': [],
        'cpu_time_per_nfe': [], 'gpu_time_per_nfe': [], 'nfe_cost_ratio': [],
        'cpu_avg_residual': [], 'gpu_avg_residual': [], 'residual_ratio': [],
    }

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                data[key].append(float(row[key]))

    return {k: np.array(v) for k, v in data.items()}


def plot_speedup(data: dict, output_dir: Path):
    """Plot 1: GPU speedup vs mesh size with crossover line."""
    fig, ax = plt.subplots(figsize=(7, 5))

    cells = data['cells']
    speedup = data['step_speedup']

    ax.semilogx(cells, speedup, 'o-', color=GPU_COLOR, linewidth=2,
                markersize=8, markerfacecolor='white', markeredgewidth=2,
                label='GPU/CPU speedup')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Parity')

    # Annotate crossover
    for i in range(len(cells)):
        ax.annotate(f'{speedup[i]:.2f}x',
                     (cells[i], speedup[i]),
                     textcoords='offset points',
                     xytext=(0, 12), ha='center', fontsize=9)

    # Shade GPU-wins region
    ax.fill_between([cells.min() * 0.5, cells.max() * 2], 1, max(speedup) * 1.3,
                     alpha=0.05, color=GPU_COLOR)
    ax.fill_between([cells.min() * 0.5, cells.max() * 2], 0, 1,
                     alpha=0.05, color=CPU_COLOR)

    ax.text(cells[0], 0.5, 'CPU faster', color=CPU_COLOR, fontsize=10,
            ha='left', style='italic')
    ax.text(cells[-1], max(speedup) * 1.1, 'GPU faster', color=GPU_COLOR,
            fontsize=10, ha='right', style='italic')

    ax.set_xlabel('Mesh cells')
    ax.set_ylabel('Speedup (CPU time / GPU time)')
    ax.set_title('GPU Speedup vs Mesh Size\n'
                 'CPU: PCG+DIC | GPU: OGLPCG+BlockJacobi (FP64)')
    ax.set_xlim(cells.min() * 0.5, cells.max() * 2)
    ax.set_ylim(0, max(speedup) * 1.3)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(ScalarFormatter())

    path = output_dir / 'scaling_speedup.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_walltime(data: dict, output_dir: Path):
    """Plot 2: Wall time per step (log-log) for CPU and GPU."""
    fig, ax = plt.subplots(figsize=(7, 5))

    cells = data['cells']

    ax.loglog(cells, data['cpu_time_per_step'], 's-', color=CPU_COLOR,
              linewidth=2, markersize=8, label='CPU (PCG+DIC)')
    ax.loglog(cells, data['gpu_time_per_step'], 'o-', color=GPU_COLOR,
              linewidth=2, markersize=8, label='GPU (OGLPCG+BJ)')

    # Reference slopes
    x_ref = np.array([cells[0], cells[-1]])
    # O(N) reference
    y_ref_n1 = data['cpu_time_per_step'][0] * (x_ref / x_ref[0])
    ax.loglog(x_ref, y_ref_n1, ':', color='gray', linewidth=1, label='O(N) ref')
    # O(N^1.5) reference
    y_ref_n15 = data['cpu_time_per_step'][0] * (x_ref / x_ref[0]) ** 1.5
    ax.loglog(x_ref, y_ref_n15, '--', color='gray', linewidth=1,
              alpha=0.5, label=r'O($N^{1.5}$) ref')

    ax.set_xlabel('Mesh cells')
    ax.set_ylabel('Wall time per timestep (s)')
    ax.set_title('Wall Time Scaling\n'
                 'maxCo = 0.5 | FP64 | tolerance 1e-6, relTol 0.1')
    ax.legend(loc='upper left')

    path = output_dir / 'scaling_walltime.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_nfe_cost(data: dict, output_dir: Path):
    """Plot 3: NFE cost ratio and iteration ratio vs mesh size."""
    fig, ax1 = plt.subplots(figsize=(7, 5))

    cells = data['cells']

    # NFE cost ratio (left axis)
    ln1 = ax1.semilogx(cells, data['nfe_cost_ratio'], 'o-', color=RATIO_COLOR,
                        linewidth=2, markersize=8, markerfacecolor='white',
                        markeredgewidth=2, label='NFE cost ratio (GPU/CPU)')
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax1.set_xlabel('Mesh cells')
    ax1.set_ylabel('NFE cost ratio (GPU time/iter) / (CPU time/iter)',
                    color=RATIO_COLOR)
    ax1.tick_params(axis='y', labelcolor=RATIO_COLOR)
    ax1.set_ylim(0, max(data['nfe_cost_ratio']) * 1.2)

    # Iteration ratio (right axis)
    ax2 = ax1.twinx()
    iter_ratio = data['gpu_iters_per_step'] / data['cpu_iters_per_step']
    ln2 = ax2.semilogx(cells, iter_ratio, 's--', color='#984ea3',
                        linewidth=2, markersize=7, label='Iteration ratio (GPU/CPU)')
    ax2.set_ylabel('Iteration ratio (GPU iters / CPU iters)', color='#984ea3')
    ax2.tick_params(axis='y', labelcolor='#984ea3')

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')

    ax1.set_title('Per-Iteration Cost and Iteration Count\n'
                  'NFE cost < 1 means GPU iterations are cheaper')

    # Annotate key points
    for i in range(len(cells)):
        ax1.annotate(f'{data["nfe_cost_ratio"][i]:.2f}',
                     (cells[i], data['nfe_cost_ratio'][i]),
                     textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8, color=RATIO_COLOR)

    path = output_dir / 'scaling_nfe_cost.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_residuals(data: dict, output_dir: Path):
    """Plot 4: Final residuals for CPU and GPU to show convergence fairness."""
    fig, ax = plt.subplots(figsize=(7, 5))

    cells = data['cells']

    ax.semilogy(cells, data['cpu_avg_residual'], 's-', color=CPU_COLOR,
                linewidth=2, markersize=8, label='CPU final residual')
    ax.semilogy(cells, data['gpu_avg_residual'], 'o-', color=GPU_COLOR,
                linewidth=2, markersize=8, label='GPU final residual')

    # Show residual ratio as bar chart on secondary axis
    ax2 = ax.twinx()
    width = [c * 0.3 for c in cells]
    ax2.bar(cells, data['residual_ratio'], width=width, alpha=0.15,
            color='orange', label='Residual ratio (GPU/CPU)')
    ax2.set_ylabel('Residual ratio (GPU/CPU)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    ax.set_xscale('log')
    ax.set_xlabel('Mesh cells')
    ax.set_ylabel('Average final residual')
    ax.set_title('Convergence Quality Comparison\n'
                 'Both solvers use tolerance 1e-6, relTol 0.1')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    path = output_dir / 'scaling_residuals.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Plot scaling benchmark results')
    parser.add_argument('--csv', required=True, help='Path to scaling_results.csv')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots (default: same as CSV)')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {csv_path}")
    data = load_scaling_csv(csv_path)
    print(f"  {len(data['cells'])} mesh sizes: {data['cells'].astype(int).tolist()}")

    print("Generating plots...")
    plot_speedup(data, output_dir)
    plot_walltime(data, output_dir)
    plot_nfe_cost(data, output_dir)
    plot_residuals(data, output_dir)
    print("Done.")


if __name__ == '__main__':
    main()
