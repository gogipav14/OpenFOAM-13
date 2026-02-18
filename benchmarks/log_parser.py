#!/usr/bin/env python3
"""
Parse OpenFOAM solver logs to extract:
- Wall-clock time per timestep
- Number of solver iterations (NFE) per pressure solve
- Initial and final residuals
- Total execution time
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SolverIteration:
    """One pressure solve within a timestep."""
    field: str          # p, p_rgh, etc.
    solver: str         # PCG, GAMG, OGLPCG, etc.
    initial_residual: float = 0.0
    final_residual: float = 0.0
    iterations: int = 0


@dataclass
class Timestep:
    """Data from a single timestep."""
    time: float = 0.0
    delta_t: float = 0.0
    courant_max: float = 0.0
    courant_mean: float = 0.0
    execution_time: float = 0.0  # cumulative
    clock_time: float = 0.0      # cumulative
    pressure_solves: list = field(default_factory=list)

    @property
    def total_pressure_iters(self) -> int:
        """Total pressure solver iterations in this timestep."""
        return sum(s.iterations for s in self.pressure_solves)

    @property
    def total_pressure_nfe(self) -> int:
        """Total NFE (matrix-vector products) for pressure.
        For PCG/CG: NFE = iterations (one SpMV per iteration).
        For GAMG: NFE ≈ iterations * levels (approximate).
        We use iterations as the consistent metric.
        """
        return self.total_pressure_iters


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics from a benchmark run."""
    case_name: str = ""
    variant: str = ""  # "cpu" or "gpu"
    solver_type: str = ""  # PCG, GAMG, OGLPCG

    # Timing
    total_wall_time: float = 0.0   # seconds
    total_exec_time: float = 0.0   # seconds
    mesh_time: float = 0.0         # seconds (from blockMesh/snappy logs)

    # Per-timestep stats
    num_timesteps: int = 0
    avg_time_per_step: float = 0.0
    min_time_per_step: float = 0.0
    max_time_per_step: float = 0.0

    # Solver iterations (NFE)
    total_pressure_iters: int = 0
    avg_iters_per_step: float = 0.0
    avg_iters_per_solve: float = 0.0
    total_pressure_solves: int = 0

    # Residuals
    avg_initial_residual: float = 0.0
    avg_final_residual: float = 0.0

    # Status
    completed: bool = False
    error_message: str = ""

    # Raw timestep data
    timesteps: list = field(default_factory=list)


def parse_solver_log(log_path: Path) -> list:
    """
    Parse an OpenFOAM solver log file and extract timestep data.

    Expected log format:
        Time = 0.01
        Courant Number mean: 0.5 max: 1.2
        smoothSolver:  Solving for Ux, ...
        GAMG:  Solving for p, Initial residual = 1, Final residual = 0.001, No Iterations 12
        ...
        ExecutionTime = 0.5 s  ClockTime = 1 s

    Returns list of Timestep objects.
    """
    if not log_path.exists():
        return []

    content = log_path.read_text(errors="replace")
    lines = content.split('\n')

    timesteps = []
    current_ts = None

    # Regex patterns
    # Time may have optional 's' suffix: "Time = 0.01s" or "Time = 0.01"
    time_pattern = re.compile(r'^Time = ([\d.eE+-]+)s?\s*$')
    courant_pattern = re.compile(
        r'Courant Number mean:\s*([\d.eE+-]+)\s+max:\s*([\d.eE+-]+)'
    )
    dt_pattern = re.compile(r'^deltaT = ([\d.eE+-]+)')
    # Match any solver name (DICPCG, GAMG, PCG, OGLPCG:FP32, etc.)
    # Formats: "DICPCG:  Solving for p, ..."
    #          "OGLPCG:FP32:  Solving for p, ..."
    solver_pattern = re.compile(
        r'([\w]+(?::[\w]+)?):\s+Solving for (\w+),\s+'
        r'Initial residual = ([\d.eE+-]+),\s+'
        r'Final residual = ([\d.eE+-]+),\s+'
        r'No Iterations (\d+)'
    )
    exectime_pattern = re.compile(
        r'ExecutionTime = ([\d.]+) s\s+ClockTime = ([\d.]+) s'
    )

    for line in lines:
        # New timestep
        m = time_pattern.match(line)
        if m:
            if current_ts is not None:
                timesteps.append(current_ts)
            current_ts = Timestep(time=float(m.group(1)))
            continue

        if current_ts is None:
            continue

        # deltaT
        m = dt_pattern.match(line)
        if m:
            current_ts.delta_t = float(m.group(1))
            continue

        # Courant number
        m = courant_pattern.search(line)
        if m:
            current_ts.courant_mean = float(m.group(1))
            current_ts.courant_max = float(m.group(2))
            continue

        # Solver iteration line (matches DICPCG, GAMG, PCG, OGLPCG, etc.)
        m = solver_pattern.search(line)
        if m:
            solve = SolverIteration(
                solver=m.group(1),
                field=m.group(2),
                initial_residual=float(m.group(3)),
                final_residual=float(m.group(4)),
                iterations=int(m.group(5)),
            )
            # Only track pressure fields
            if solve.field in ('p', 'p_rgh', 'pcorr', 'pa'):
                current_ts.pressure_solves.append(solve)
            continue

        # Execution time
        m = exectime_pattern.search(line)
        if m:
            current_ts.execution_time = float(m.group(1))
            current_ts.clock_time = float(m.group(2))
            continue

    # Don't forget the last timestep
    if current_ts is not None:
        timesteps.append(current_ts)

    return timesteps


def compute_metrics(
    timesteps: list,
    case_name: str = "",
    variant: str = "",
    warmup_steps: int = 5,
) -> BenchmarkMetrics:
    """
    Compute aggregated benchmark metrics from timestep data.

    Args:
        timesteps: List of Timestep objects from log parsing
        case_name: Tutorial case name
        variant: "cpu" or "gpu"
        warmup_steps: Number of initial timesteps to exclude from timing stats
    """
    metrics = BenchmarkMetrics(
        case_name=case_name,
        variant=variant,
        timesteps=timesteps,
    )

    if not timesteps:
        metrics.error_message = "No timesteps parsed from log"
        return metrics

    metrics.completed = True
    metrics.num_timesteps = len(timesteps)

    # Total times — prefer execution_time (float) over clock_time (integer)
    # for precision on fast cases
    metrics.total_exec_time = timesteps[-1].execution_time
    metrics.total_wall_time = (
        timesteps[-1].clock_time
        if timesteps[-1].clock_time > 0
        else timesteps[-1].execution_time
    )

    # Determine solver type from first pressure solve
    for ts in timesteps:
        if ts.pressure_solves:
            metrics.solver_type = ts.pressure_solves[0].solver
            break

    # Compute per-timestep times using execution_time (higher precision than
    # integer ClockTime, especially for small/fast cases)
    step_times = []
    for i in range(1, len(timesteps)):
        dt = timesteps[i].execution_time - timesteps[i - 1].execution_time
        if dt >= 0:
            step_times.append(dt)

    # Skip warmup steps for statistics
    measured_steps = step_times[warmup_steps:] if len(step_times) > warmup_steps else step_times

    if measured_steps:
        metrics.avg_time_per_step = sum(measured_steps) / len(measured_steps)
        metrics.min_time_per_step = min(measured_steps)
        metrics.max_time_per_step = max(measured_steps)

    # Pressure solver iterations
    all_solves = []
    for ts in timesteps:
        all_solves.extend(ts.pressure_solves)

    metrics.total_pressure_solves = len(all_solves)
    metrics.total_pressure_iters = sum(s.iterations for s in all_solves)

    if all_solves:
        metrics.avg_iters_per_solve = (
            metrics.total_pressure_iters / len(all_solves)
        )

    if timesteps:
        metrics.avg_iters_per_step = (
            metrics.total_pressure_iters / len(timesteps)
        )

    # Residuals
    initial_residuals = [s.initial_residual for s in all_solves if s.initial_residual > 0]
    final_residuals = [s.final_residual for s in all_solves if s.final_residual > 0]

    if initial_residuals:
        metrics.avg_initial_residual = sum(initial_residuals) / len(initial_residuals)
    if final_residuals:
        metrics.avg_final_residual = sum(final_residuals) / len(final_residuals)

    return metrics


def parse_mesh_log(log_path: Path) -> float:
    """Parse a mesh generation log and return wall-clock time in seconds."""
    if not log_path.exists():
        return 0.0

    content = log_path.read_text(errors="replace")
    m = re.search(r'ExecutionTime = ([\d.]+) s', content)
    if m:
        return float(m.group(1))

    # Try ClockTime
    m = re.search(r'ClockTime = ([\d.]+) s', content)
    if m:
        return float(m.group(1))

    return 0.0


def find_solver_log(case_path: Path) -> Optional[Path]:
    """
    Find the solver log file in a case directory.
    OpenFOAM creates logs as log.foamRun, log.icoFoam, etc.
    """
    # Check for common log names
    log_names = [
        "log.foamRun",
        "log.icoFoam",
        "log.simpleFoam",
        "log.pimpleFoam",
        "log.pisoFoam",
        "log.buoyantSimpleFoam",
        "log.buoyantPimpleFoam",
        "log.interFoam",
        "log.rhoPimpleFoam",
        "log.rhoSimpleFoam",
        "log.reactingFoam",
        "log.potentialFoam",
        "log.sonicFoam",
    ]

    for name in log_names:
        log_path = case_path / name
        if log_path.exists():
            return log_path

    # Fallback: find any log.* file that contains "Time ="
    for log_file in sorted(case_path.glob("log.*")):
        if log_file.is_file():
            try:
                head = log_file.read_text(errors="replace")[:2000]
                if "Time =" in head:
                    return log_file
            except Exception:
                continue

    return None


def format_metrics_table(metrics_list: list) -> str:
    """Format a list of BenchmarkMetrics as a readable table."""
    if not metrics_list:
        return "No metrics to display."

    header = (
        f"{'Case':<40} {'Variant':<6} {'Solver':<8} "
        f"{'Steps':>6} {'Wall(s)':>8} {'s/step':>8} "
        f"{'TotIters':>8} {'Iter/step':>9} {'Avg Res':>10}"
    )
    separator = "─" * len(header)

    lines = [header, separator]
    for m in metrics_list:
        name = m.case_name[:40] if len(m.case_name) > 40 else m.case_name
        lines.append(
            f"{name:<40} {m.variant:<6} {m.solver_type:<8} "
            f"{m.num_timesteps:>6} {m.total_wall_time:>8.2f} "
            f"{m.avg_time_per_step:>8.4f} "
            f"{m.total_pressure_iters:>8} "
            f"{m.avg_iters_per_step:>9.1f} "
            f"{m.avg_final_residual:>10.2e}"
        )

    return '\n'.join(lines)
