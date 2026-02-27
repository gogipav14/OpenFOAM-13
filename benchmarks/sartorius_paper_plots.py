#!/usr/bin/env python3
"""Generate paper-quality figures for Sartorius 3D stirred-tank benchmarks.

Produces two figures for Paper 1:
  1. sartorius_scaling.pdf - Scaling comparison (CPU vs GPU)
  2. cache_sweep.pdf - Cache interval sensitivity at 5M cells

Usage:
  python3 sartorius_paper_plots.py
"""

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Consistent style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

RESULTS_DIR = Path(__file__).parent / 'sartorius_scaling_results'
CACHE_DIR = Path(__file__).parent / 'sartorius_cache_results'
FIG_DIR = Path(__file__).parent / 'sartorius_figures'
FIG_DIR.mkdir(exist_ok=True)


def read_scaling_csv():
    """Read scaling results CSV."""
    data = {}
    with open(RESULTS_DIR / 'scaling_results.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            solver = row['solver']
            if solver not in data:
                data[solver] = {'cells': [], 'time': [], 'p_iters': [], 'pf_iters': []}
            data[solver]['cells'].append(int(row['ncells']))
            data[solver]['time'].append(float(row['exec_time_s']))
            data[solver]['p_iters'].append(float(row['avg_p_iters']))
            data[solver]['pf_iters'].append(float(row['avg_pFinal_iters']))
    return data


def read_cache_csv():
    """Read cache sweep CSV."""
    data = {}
    with open(CACHE_DIR / 'cache_sweep.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ci = int(row['cache_interval'])
            precond = row['preconditioner']
            key = f"{precond}_ci{ci}"
            data[key] = {
                'ci': ci,
                'precond': precond,
                'p_iters': float(row['avg_p_iters']) if row['avg_p_iters'] not in ('FATAL', 'N/A') else None,
                'pf_iters': float(row['avg_pFinal_iters']) if row['avg_pFinal_iters'] not in ('FATAL', 'N/A') else None,
                'time': float(row['exec_time_s']) if row['exec_time_s'] not in ('FATAL', 'N/A') else None,
                'avg_ms': float(row['avg_p_solve_ms']) if row['avg_p_solve_ms'] not in ('FATAL', 'N/A') else None,
            }
    return data


def plot_scaling():
    """Figure: Scaling comparison CPU vs GPU.

    Note: The 5M GPU results from the original scaling run were contaminated
    by GPU contention. We substitute clean data from the cache sweep (ci=10).
    """
    data = read_scaling_csv()

    # Substitute contaminated 5M GPU results with clean cache sweep data
    cache = read_cache_csv()
    bj_clean = next((v for v in cache.values() if v['precond'] == 'blockJacobi' and v['ci'] == 10), None)
    ilu_clean = next((v for v in cache.values() if v['precond'] == 'ILU' and v['ci'] == 10), None)
    cpu_clean = next((v for v in cache.values() if v['precond'] == 'cpuGAMG'), None)
    if bj_clean and 'gpuAMG_BJ' in data:
        data['gpuAMG_BJ']['time'][-1] = bj_clean['time']
        data['gpuAMG_BJ']['p_iters'][-1] = bj_clean['p_iters']
    if ilu_clean and 'gpuAMG_ILU' in data:
        data['gpuAMG_ILU']['time'][-1] = ilu_clean['time']
        data['gpuAMG_ILU']['p_iters'][-1] = ilu_clean['p_iters']
    if cpu_clean and 'cpuGAMG' in data:
        data['cpuGAMG']['time'][-1] = cpu_clean['time']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    colors = {'cpuGAMG': '#2196F3', 'gpuAMG_BJ': '#FF5722', 'gpuAMG_ILU': '#4CAF50'}
    labels = {'cpuGAMG': 'CPU GAMG', 'gpuAMG_BJ': 'GPU AMG+BJ', 'gpuAMG_ILU': 'GPU AMG+ILU'}
    markers = {'cpuGAMG': 's', 'gpuAMG_BJ': 'o', 'gpuAMG_ILU': '^'}

    # Panel 1: Total execution time
    ax = axes[0]
    for solver in ['cpuGAMG', 'gpuAMG_BJ', 'gpuAMG_ILU']:
        d = data[solver]
        cells_M = [c / 1e6 for c in d['cells']]
        ax.semilogy(cells_M, d['time'], f'-{markers[solver]}', color=colors[solver],
                    label=labels[solver], markersize=6, linewidth=1.5)
    ax.set_xlabel('Cells (millions)')
    ax.set_ylabel('Execution time (s)')
    ax.set_title('Total wall time (10 timesteps)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: GPU/CPU ratio
    ax = axes[1]
    cpu = data['cpuGAMG']
    for solver in ['gpuAMG_BJ', 'gpuAMG_ILU']:
        d = data[solver]
        cells_M = [c / 1e6 for c in d['cells']]
        ratio = [g / c for g, c in zip(d['time'], cpu['time'])]
        ax.plot(cells_M, ratio, f'-{markers[solver]}', color=colors[solver],
                label=labels[solver], markersize=6, linewidth=1.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlabel('Cells (millions)')
    ax.set_ylabel('GPU / CPU time ratio')
    ax.set_title('GPU/CPU ratio (<1 = GPU faster)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Pressure iterations
    ax = axes[2]
    for solver in ['cpuGAMG', 'gpuAMG_BJ', 'gpuAMG_ILU']:
        d = data[solver]
        cells_M = [c / 1e6 for c in d['cells']]
        ax.plot(cells_M, d['p_iters'], f'-{markers[solver]}', color=colors[solver],
                label=labels[solver], markersize=6, linewidth=1.5)
    ax.set_xlabel('Cells (millions)')
    ax.set_ylabel('Avg. pressure iterations')
    ax.set_title('Pressure iteration count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Sartorius 3D Stirred Tank: Scaling Study (RTX 5060)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'sartorius_scaling.png')
    fig.savefig(FIG_DIR / 'sartorius_scaling.pdf')
    print(f'  Saved scaling figure')
    plt.close()


def plot_cache_sweep():
    """Figure: Cache interval sensitivity at 5M cells."""
    data = read_cache_csv()

    # Separate BJ, ILU, and CPU
    bj_entries = sorted(
        [v for v in data.values() if v['precond'] == 'blockJacobi' and v['time'] is not None],
        key=lambda x: x['ci']
    )
    ilu_entries = sorted(
        [v for v in data.values() if v['precond'] == 'ILU' and v['time'] is not None],
        key=lambda x: x['ci']
    )
    cpu_entry = next((v for v in data.values() if v['precond'] == 'cpuGAMG'), None)

    if not bj_entries:
        print('  No BJ cache sweep data yet')
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: Execution time vs cache interval
    ax = axes[0]
    bj_cis = [v['ci'] for v in bj_entries]
    bj_times = [v['time'] for v in bj_entries]
    ax.plot(bj_cis, bj_times, '-o', color='#FF5722', label='GPU AMG+BJ', markersize=8, linewidth=2)

    if ilu_entries:
        ilu_cis = [v['ci'] for v in ilu_entries]
        ilu_times = [v['time'] for v in ilu_entries]
        ax.plot(ilu_cis, ilu_times, '-^', color='#4CAF50', label='GPU AMG+ILU', markersize=8, linewidth=2)

    if cpu_entry and cpu_entry['time']:
        ax.axhline(y=cpu_entry['time'], color='#2196F3', linestyle='--', linewidth=1.5,
                   label=f'CPU GAMG ({cpu_entry["time"]:.0f}s)')

    ax.set_xlabel('mgCacheInterval')
    ax.set_ylabel('Execution time (s)')
    ax.set_title('Wall time vs. cache interval')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Pressure iterations vs cache interval
    ax = axes[1]
    bj_piters = [v['p_iters'] for v in bj_entries]
    ax.plot(bj_cis, bj_piters, '-o', color='#FF5722', label='GPU AMG+BJ', markersize=8, linewidth=2)

    if ilu_entries:
        ilu_piters = [v['p_iters'] for v in ilu_entries]
        ax.plot(ilu_cis, ilu_piters, '-^', color='#4CAF50', label='GPU AMG+ILU', markersize=8, linewidth=2)

    if cpu_entry and cpu_entry['p_iters']:
        ax.axhline(y=cpu_entry['p_iters'], color='#2196F3', linestyle='--', linewidth=1.5,
                   label='CPU GAMG')

    ax.set_xlabel('mgCacheInterval')
    ax.set_ylabel('Avg. pressure iterations')
    ax.set_title('Iteration degradation with stale hierarchy')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Cache Interval Sensitivity: Sartorius 5M cells (RTX 5060)', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'cache_sweep.png')
    fig.savefig(FIG_DIR / 'cache_sweep.pdf')
    print(f'  Saved cache sweep figure')
    plt.close()


if __name__ == '__main__':
    print('Generating Sartorius paper figures...')
    try:
        plot_scaling()
    except Exception as e:
        print(f'  Scaling plot failed: {e}')
    try:
        plot_cache_sweep()
    except Exception as e:
        print(f'  Cache sweep plot failed: {e}')
    print(f'All figures saved to {FIG_DIR}/')
