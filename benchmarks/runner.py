#!/usr/bin/env python3
"""
Run OpenFOAM benchmark cases, supporting both native and Docker execution.
"""

import os
import subprocess
import shutil
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from log_parser import (
    parse_solver_log,
    compute_metrics,
    find_solver_log,
    parse_mesh_log,
    BenchmarkMetrics,
)


@dataclass
class RunConfig:
    """Configuration for running benchmarks."""
    # Docker settings
    use_docker: bool = False
    docker_image: str = "mixfoam:latest"
    gpu_runtime: bool = True  # Use --runtime=nvidia / --gpus all

    # Dev container settings (persistent container, docker exec instead of run)
    use_dev_container: bool = False
    dev_container_name: str = "ogl-dev"

    # Native settings
    openfoam_dir: str = ""  # WM_PROJECT_DIR

    # Execution settings
    timeout: int = 600  # seconds per case
    dry_run: bool = False
    verbose: bool = False


def run_case_native(
    case_path: Path,
    run_config: RunConfig,
) -> BenchmarkMetrics:
    """
    Run an OpenFOAM case natively (assumes OpenFOAM is sourced).
    """
    case_name = case_path.name

    # Clean any previous results
    _clean_case(case_path)

    # Run mesh generation
    mesh_time = 0.0
    blockmesh = case_path / "system" / "blockMeshDict"
    if blockmesh.exists():
        mesh_time += _run_openfoam_cmd(
            case_path, "blockMesh", run_config
        )

    toposet = case_path / "system" / "topoSetDict"
    if toposet.exists():
        _run_openfoam_cmd(case_path, "topoSet", run_config)

    # Run solver
    solver_cmd = _detect_solver_command(case_path)

    if run_config.dry_run:
        print(f"  [DRY RUN] Would run: {solver_cmd} in {case_path}")
        return BenchmarkMetrics(case_name=case_name, variant="dry_run")

    start_time = time.time()
    try:
        result = subprocess.run(
            [solver_cmd],
            cwd=str(case_path),
            capture_output=True,
            text=True,
            timeout=run_config.timeout,
        )
        elapsed = time.time() - start_time

        # Write the log
        log_path = case_path / f"log.{solver_cmd}"
        log_path.write_text(result.stdout + result.stderr)

        if run_config.verbose:
            print(f"  Solver completed in {elapsed:.1f}s (exit code: {result.returncode})")

    except subprocess.TimeoutExpired:
        return BenchmarkMetrics(
            case_name=case_name,
            error_message=f"Timeout after {run_config.timeout}s",
        )
    except FileNotFoundError:
        return BenchmarkMetrics(
            case_name=case_name,
            error_message=f"Solver command not found: {solver_cmd}",
        )

    # Parse results
    log_file = find_solver_log(case_path)
    if log_file:
        timesteps = parse_solver_log(log_file)
        metrics = compute_metrics(timesteps, case_name=case_name)
        metrics.mesh_time = mesh_time
        return metrics

    return BenchmarkMetrics(
        case_name=case_name,
        error_message="No solver log found",
    )


def run_case_docker(
    case_path: Path,
    run_config: RunConfig,
    variant: str = "cpu",
) -> BenchmarkMetrics:
    """
    Run an OpenFOAM case inside a Docker container.

    The case directory is mounted as a volume.
    """
    case_name = case_path.name

    # Clean any previous results
    _clean_case(case_path)

    # Build Docker command
    docker_cmd = ["docker", "run", "--rm"]

    # GPU support
    if variant == "gpu" and run_config.gpu_runtime:
        docker_cmd.extend(["--gpus", "all"])

    # Mount the case directory (must be absolute for Docker volume mount)
    abs_case_path = case_path.resolve()
    docker_cmd.extend([
        "-v", f"{abs_case_path}:/benchmark/case",
        "-w", "/benchmark/case",
        run_config.docker_image,
    ])

    # The command to run inside the container
    # First source OpenFOAM, then run mesh + solver
    inner_script = _build_inner_script(case_path, host_uid=os.getuid())
    docker_cmd.extend(["bash", "-c", inner_script])

    if run_config.dry_run:
        print(f"  [DRY RUN] Docker: {' '.join(docker_cmd[:10])}...")
        return BenchmarkMetrics(case_name=case_name, variant="dry_run")

    if run_config.verbose:
        print(f"  Running Docker ({variant})...")

    start_time = time.time()
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=run_config.timeout,
        )
        elapsed = time.time() - start_time

        if run_config.verbose:
            print(f"  Docker completed in {elapsed:.1f}s (exit code: {result.returncode})")
            if result.returncode != 0 and result.stderr:
                print(f"  stderr: {result.stderr[:200]}")

    except subprocess.TimeoutExpired:
        return BenchmarkMetrics(
            case_name=case_name,
            variant=variant,
            error_message=f"Docker timeout after {run_config.timeout}s",
        )

    # Parse results from the mounted case directory
    log_file = find_solver_log(case_path)
    if log_file:
        timesteps = parse_solver_log(log_file)
        metrics = compute_metrics(timesteps, case_name=case_name, variant=variant)

        # Also parse mesh log if available
        mesh_log = case_path / "log.blockMesh"
        if mesh_log.exists():
            metrics.mesh_time = parse_mesh_log(mesh_log)

        # Use Docker elapsed time as wall time if OpenFOAM didn't report it
        if metrics.total_wall_time == 0 and elapsed > 0:
            solver_wall = elapsed - metrics.mesh_time
            metrics.total_wall_time = max(solver_wall, 0)
            metrics.total_exec_time = metrics.total_wall_time
            # Recompute per-step from total if per-step deltas were 0
            if metrics.avg_time_per_step == 0 and metrics.num_timesteps > 0:
                metrics.avg_time_per_step = (
                    metrics.total_wall_time / metrics.num_timesteps
                )

        return metrics

    return BenchmarkMetrics(
        case_name=case_name,
        variant=variant,
        error_message="No solver log found after Docker run",
    )


def run_case_dev_container(
    case_path: Path,
    run_config: RunConfig,
    variant: str = "cpu",
) -> BenchmarkMetrics:
    """
    Run an OpenFOAM case via docker exec on a persistent dev container.

    Unlike run_case_docker (docker run --rm), this uses an already-running
    container with the latest compiled binaries. Much faster iteration cycle
    since no image rebuild is needed — just wmake + exec.
    """
    case_name = case_path.name
    container = run_config.dev_container_name

    # Verify container is running
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format={{.State.Running}}", container],
            capture_output=True, text=True,
        )
        if "true" not in result.stdout:
            return BenchmarkMetrics(
                case_name=case_name,
                variant=variant,
                error_message=f"Dev container '{container}' is not running. "
                              f"Run './dev.sh start' first.",
            )
    except FileNotFoundError:
        return BenchmarkMetrics(
            case_name=case_name,
            variant=variant,
            error_message="Docker not found",
        )

    # Clean any previous results
    _clean_case(case_path)

    # Copy the case into the container (docker exec can't mount volumes)
    abs_case_path = case_path.resolve()
    container_case_dir = f"/tmp/bench/{case_name}"

    # Build the inner script
    inner_script = _build_inner_script_dev(
        case_path, container_case_dir, host_uid=os.getuid()
    )

    if run_config.dry_run:
        print(f"  [DRY RUN] docker exec {container}: {inner_script[:80]}...")
        return BenchmarkMetrics(case_name=case_name, variant="dry_run")

    if run_config.verbose:
        print(f"  Running in dev container ({variant})...")

    # Copy case into container
    subprocess.run(
        ["docker", "exec", container, "bash", "-c",
         f"rm -rf {container_case_dir} && mkdir -p {container_case_dir}"],
        capture_output=True,
    )
    subprocess.run(
        ["docker", "cp", f"{abs_case_path}/.", f"{container}:{container_case_dir}"],
        capture_output=True,
    )

    start_time = time.time()
    try:
        result = subprocess.run(
            ["docker", "exec", container, "bash", "-c", inner_script],
            capture_output=True,
            text=True,
            timeout=run_config.timeout,
        )
        elapsed = time.time() - start_time

        if run_config.verbose:
            print(f"  Dev container completed in {elapsed:.1f}s "
                  f"(exit code: {result.returncode})")
            if result.returncode != 0 and result.stderr:
                print(f"  stderr: {result.stderr[:200]}")

    except subprocess.TimeoutExpired:
        return BenchmarkMetrics(
            case_name=case_name,
            variant=variant,
            error_message=f"Dev container timeout after {run_config.timeout}s",
        )

    # Copy results back from container
    subprocess.run(
        ["docker", "cp", f"{container}:{container_case_dir}/.", str(abs_case_path)],
        capture_output=True,
    )

    # Parse results
    log_file = find_solver_log(case_path)
    if log_file:
        timesteps = parse_solver_log(log_file)
        metrics = compute_metrics(timesteps, case_name=case_name, variant=variant)

        mesh_log = case_path / "log.blockMesh"
        if mesh_log.exists():
            metrics.mesh_time = parse_mesh_log(mesh_log)

        if metrics.total_wall_time == 0 and elapsed > 0:
            solver_wall = elapsed - metrics.mesh_time
            metrics.total_wall_time = max(solver_wall, 0)
            metrics.total_exec_time = metrics.total_wall_time
            if metrics.avg_time_per_step == 0 and metrics.num_timesteps > 0:
                metrics.avg_time_per_step = (
                    metrics.total_wall_time / metrics.num_timesteps
                )

        return metrics

    return BenchmarkMetrics(
        case_name=case_name,
        variant=variant,
        error_message="No solver log found after dev container run",
    )


def _build_inner_script_dev(
    case_path: Path, container_case_dir: str, host_uid: int = None,
) -> str:
    """Build the shell script for docker exec in the dev container."""
    parts = [
        "export USER=${USER:-root}",
        "export FOAM_INST_DIR=/opt",
        "source /opt/OpenFOAM-13/etc/bashrc 2>/dev/null || true",
        "export GINKGO_ROOT=/opt/ginkgo",
        f"cd {container_case_dir}",
    ]

    if (case_path / "system" / "blockMeshDict").exists():
        parts.append("blockMesh > log.blockMesh 2>&1")

    if (case_path / "system" / "topoSetDict").exists():
        parts.append("topoSet > log.topoSet 2>&1")

    if (case_path / "system" / "snappyHexMeshDict").exists():
        parts.append("snappyHexMesh -overwrite > log.snappyHexMesh 2>&1")

    solver_cmd = _detect_solver_command(case_path)
    parts.append(f"{solver_cmd} > log.{solver_cmd} 2>&1")

    return " && ".join(parts)


def _build_inner_script(case_path: Path, host_uid: int = None) -> str:
    """Build the shell script to run inside the Docker container."""
    parts = [
        # Set USER for OpenFOAM's WM_PROJECT_USER_DIR path resolution
        # (needed so libOGL.so is found in $HOME/OpenFOAM/$USER-13/...)
        "export USER=${USER:-root}",
        "source /opt/OpenFOAM-13/etc/bashrc 2>/dev/null || source /opt/openfoam13/etc/bashrc 2>/dev/null || true",
    ]

    # Check for mesh requirements
    if (case_path / "system" / "blockMeshDict").exists():
        parts.append("blockMesh > log.blockMesh 2>&1")

    if (case_path / "system" / "topoSetDict").exists():
        parts.append("topoSet > log.topoSet 2>&1")

    if (case_path / "system" / "snappyHexMeshDict").exists():
        parts.append("snappyHexMesh -overwrite > log.snappyHexMesh 2>&1")

    # Determine solver
    solver_cmd = _detect_solver_command(case_path)
    parts.append(f"{solver_cmd} > log.{solver_cmd} 2>&1")

    script = " && ".join(parts)

    # Fix file ownership so host user can read/delete output.
    # Use ';' so chown runs even if the solver fails.
    if host_uid is not None:
        script += f" ; chown -R {host_uid}:{host_uid} /benchmark/case"

    return script


def _detect_solver_command(case_path: Path) -> str:
    """Detect the appropriate solver command from controlDict."""
    controldict = case_path / "system" / "controlDict"
    if not controldict.exists():
        return "foamRun"

    content = controldict.read_text(errors="replace")

    # Check for legacy application keyword
    import re
    app_match = re.search(r'application\s+(\w+)\s*;', content)
    if app_match:
        return app_match.group(1)

    # Modern OF13: "solver" keyword means use foamRun
    solver_match = re.search(r'^\s*solver\s+\w+\s*;', content, re.MULTILINE)
    if solver_match:
        return "foamRun"

    return "foamRun"


def _clean_case(case_path: Path):
    """Clean a case directory of previous results."""
    # Remove time directories (except 0)
    for item in case_path.iterdir():
        if item.is_dir():
            try:
                t = float(item.name)
                if t > 0:
                    shutil.rmtree(item)
            except ValueError:
                continue

    # Remove processor directories
    for proc_dir in case_path.glob("processor*"):
        if proc_dir.is_dir():
            shutil.rmtree(proc_dir)

    # Remove log files
    for log_file in case_path.glob("log.*"):
        log_file.unlink()

    # Remove postProcessing
    pp_dir = case_path / "postProcessing"
    if pp_dir.exists():
        shutil.rmtree(pp_dir)
