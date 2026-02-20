"""
Execute the OpenFOAM simulation pipeline.

Steps:
  1. blockMesh — generate background hex mesh
  2. snappyHexMesh — cut impeller/baffle geometry (if STLs present)
  3. createNonConformalCouples — set up NCC rotating interfaces
  4. setFields — initialize tracer at bottom
  5. foamRun — run the solver
"""

import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional


def _check_openfoam():
    """Check if OpenFOAM is available."""
    if shutil.which("blockMesh") is None:
        print("ERROR: OpenFOAM not found in PATH.")
        print("  Source OpenFOAM environment first:")
        print("  source /opt/openfoam13/etc/bashrc")
        return False
    return True


def _run_command(cmd: str, case_dir: str, log_name: Optional[str] = None):
    """Run a command and log output."""
    log_file = Path(case_dir) / f"log.{log_name or cmd.split()[0]}"

    print(f"  Running: {cmd}")
    with open(log_file, "w") as log:
        proc = subprocess.run(
            cmd, shell=True, cwd=case_dir,
            stdout=log, stderr=subprocess.STDOUT,
        )

    if proc.returncode != 0:
        print(f"  ERROR: {cmd} failed (exit code {proc.returncode})")
        print(f"  Check log: {log_file}")
        # Print last 20 lines of log
        lines = log_file.read_text().splitlines()
        for line in lines[-20:]:
            print(f"    {line}")
        return False

    print(f"  OK ({log_file.name})")
    return True


def run_case(case_dir: str, parallel: bool = False, n_procs: int = 4) -> bool:
    """
    Run the full simulation pipeline.

    Args:
        case_dir: Path to the OpenFOAM case directory
        parallel: Whether to run in parallel (multi-GPU)
        n_procs: Number of processors for parallel run

    Returns:
        True if all steps completed successfully
    """
    if not _check_openfoam():
        return False

    case = Path(case_dir)
    if not case.exists():
        print(f"ERROR: Case directory not found: {case}")
        return False

    print(f"\nRunning simulation: {case}")
    print("-" * 50)

    # Step 1: blockMesh
    if not _run_command("blockMesh", str(case)):
        return False

    # Step 2: snappyHexMesh (if snappyHexMeshDict exists)
    if (case / "system" / "snappyHexMeshDict").exists():
        if not _run_command("snappyHexMesh -overwrite", str(case)):
            return False

    # Step 3: createNonConformalCouples
    if not _run_command("createNonConformalCouples", str(case)):
        return False

    # Step 4: setFields
    if not _run_command("setFields", str(case)):
        return False

    # Step 5: Run solver
    if parallel:
        if not _run_command("decomposePar", str(case)):
            return False
        if not _run_command(
            f"mpirun -np {n_procs} foamRun -parallel",
            str(case), log_name="foamRun"
        ):
            return False
        if not _run_command("reconstructPar", str(case)):
            return False
    else:
        if not _run_command("foamRun", str(case)):
            return False

    print("-" * 50)
    print("Simulation completed successfully!")
    return True
