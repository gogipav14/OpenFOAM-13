#!/usr/bin/env python3
"""
Modify OpenFOAM tutorial cases for benchmarking:
- CPU baseline: keep original solver (or switch GAMG → PCG for fair comparison)
- GPU variant: replace pressure solver with OGLPCG
- Control timesteps for consistent benchmarking
"""

import re
import shutil
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    # Number of timesteps to run (0 = use original)
    max_timesteps: int = 50
    # Whether to force serial (no decomposePar)
    force_serial: bool = True
    # GPU solver settings
    precision_policy: str = "FP32"  # FP32, FP64, MIXED
    iterative_refinement: bool = True
    max_refine_iters: int = 3
    inner_tolerance: float = 1e-4
    cache_structure: bool = True
    cache_values: bool = False
    debug_level: int = 0
    # Preconditioner settings
    preconditioner: str = "Jacobi"  # Jacobi, blockJacobi, ISAI, FFT, fftBlockJacobi
    block_size: int = 4             # Block Jacobi block size
    isai_sparsity_power: int = 1    # ISAI sparsity power (1 or 2)
    # FFT preconditioner settings (required when preconditioner = FFT or fftBlockJacobi)
    fft_dimensions: tuple = None    # (nx, ny, nz) grid dimensions
    mesh_spacing: tuple = None      # (dx, dy, dz) cell spacing


# Template for OGLPCG solver entry in fvSolution
OGLPCG_TEMPLATE = """    {{
        solver          OGLPCG;
        tolerance       {tolerance};
        relTol          {relTol};
        OGLCoeffs
        {{
            precisionPolicy     {precisionPolicy};
            iterativeRefinement {iterativeRefinement};
            maxRefineIters      {maxRefineIters};
            innerTolerance      {innerTolerance};
            cacheStructure      {cacheStructure};
            cacheValues         {cacheValues};
            debug               {debug};
            preconditioner      {preconditioner};
            blockSize           {blockSize};
            isaiSparsityPower   {isaiSparsityPower};{fftEntries}
        }}
    }}"""

# Template for CPU PCG solver (to replace GAMG for fair comparison)
CPU_PCG_TEMPLATE = """    {{
        solver          PCG;
        preconditioner  DIC;
        tolerance       {tolerance};
        relTol          {relTol};
    }}"""


def prepare_case(
    src_path: Path,
    dst_path: Path,
    variant: str,
    config: BenchmarkConfig = None,
) -> Path:
    """
    Copy a tutorial case and modify it for benchmarking.

    Args:
        src_path: Original tutorial case directory
        dst_path: Destination directory for the modified case
        variant: "cpu" for baseline or "gpu" for OGLPCG
        config: Benchmark configuration

    Returns:
        Path to the prepared case directory
    """
    if config is None:
        config = BenchmarkConfig()

    # Copy the case directory
    if dst_path.exists():
        shutil.rmtree(dst_path)
    shutil.copytree(src_path, dst_path, symlinks=True)

    # Modify fvSolution
    modify_fvsolution(dst_path, variant, config)

    # Modify controlDict
    modify_controldict(dst_path, variant, config)

    # Remove decomposePar if forcing serial
    if config.force_serial:
        decompose = dst_path / "system" / "decomposeParDict"
        if decompose.exists():
            decompose.unlink()

    return dst_path


def modify_fvsolution(case_path: Path, variant: str, config: BenchmarkConfig):
    """Modify fvSolution for the given variant."""
    fvsolution = case_path / "system" / "fvSolution"
    if not fvsolution.exists():
        return

    content = fvsolution.read_text()

    if variant == "gpu":
        content = _replace_pressure_solver_gpu(content, config)
    elif variant == "cpu_pcg":
        # Force PCG+DIC for fair Krylov-vs-Krylov comparison
        content = _replace_pressure_solver_cpu(content, config)
    elif variant == "cpu":
        # CPU baseline: keep original solver exactly as-is (GAMG, PCG, etc.)
        # This ensures the CPU run is identical to vanilla OpenFOAM behavior.
        pass

    fvsolution.write_text(content)


def _replace_pressure_solver_gpu(content: str, config: BenchmarkConfig) -> str:
    """Replace pressure solver entries with OGLPCG."""
    # Find and replace pressure solver blocks for p, p_rgh, pFinal, p_rghFinal
    pressure_fields = [
        'p_rgh', 'p', 'pcorr', 'pa',
        'p_rghFinal', 'pFinal', 'pcorrFinal', 'paFinal',
    ]

    # Build FFT entries if needed
    fft_entries = ""
    if config.fft_dimensions and config.mesh_spacing:
        nx, ny, nz = config.fft_dimensions
        dx, dy, dz = config.mesh_spacing
        fft_entries = (
            f"\n            fftDimensions     ({nx} {ny} {nz});"
            f"\n            meshSpacing       ({dx} {dy} {dz});"
        )

    for pfield in pressure_fields:
        content = _replace_solver_block(
            content, pfield,
            OGLPCG_TEMPLATE.format(
                tolerance=1e-6,
                relTol=0.1 if "Final" not in pfield else 0,
                precisionPolicy=config.precision_policy,
                iterativeRefinement="on" if config.iterative_refinement else "off",
                maxRefineIters=config.max_refine_iters,
                innerTolerance=config.inner_tolerance,
                cacheStructure="true" if config.cache_structure else "false",
                cacheValues="true" if config.cache_values else "false",
                debug=config.debug_level,
                preconditioner=config.preconditioner,
                blockSize=config.block_size,
                isaiSparsityPower=config.isai_sparsity_power,
                fftEntries=fft_entries,
            )
        )

    return content


def _replace_pressure_solver_cpu(content: str, config: BenchmarkConfig) -> str:
    """Replace GAMG pressure solver with PCG+DIC for fair comparison."""
    pressure_fields = [
        'p_rgh', 'p', 'pcorr', 'pa',
        'p_rghFinal', 'pFinal', 'pcorrFinal', 'paFinal',
    ]

    for pfield in pressure_fields:
        # Only replace GAMG, leave PCG as-is
        content = _replace_solver_block_if_gamg(
            content, pfield,
            CPU_PCG_TEMPLATE.format(
                tolerance=1e-6,
                relTol=0.1 if "Final" not in pfield else 0,
            )
        )

    return content


def _replace_solver_block(content: str, field_name: str, replacement: str) -> str:
    """Replace a solver block for a given field name."""
    # Handle both quoted and unquoted field names
    # Pattern: field_name { ... }
    # Need to handle nested braces

    patterns = [
        # Quoted: "p_rgh"
        rf'("?{re.escape(field_name)}"?)\s*\{{',
        # With $reference
        rf'("?{re.escape(field_name)}"?)\s*\{{',
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            start = match.start()
            # Find the matching closing brace
            brace_start = content.index('{', start)
            end = _find_matching_brace(content, brace_start)
            if end > 0:
                # Replace the entire block
                field_label = match.group(1)
                content = (
                    content[:start]
                    + field_label + "\n"
                    + replacement + "\n"
                    + content[end + 1:]
                )
                break

    return content


def _replace_solver_block_if_gamg(
    content: str, field_name: str, replacement: str
) -> str:
    """Replace solver block only if it uses GAMG."""
    patterns = [
        rf'("?{re.escape(field_name)}"?)\s*\{{',
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            start = match.start()
            brace_start = content.index('{', start)
            end = _find_matching_brace(content, brace_start)
            if end > 0:
                block = content[brace_start:end + 1]
                if 'GAMG' in block:
                    field_label = match.group(1)
                    content = (
                        content[:start]
                        + field_label + "\n"
                        + replacement + "\n"
                        + content[end + 1:]
                    )
                break

    return content


def _find_matching_brace(text: str, start: int) -> int:
    """Find the position of the matching closing brace."""
    if text[start] != '{':
        return -1

    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1


def modify_controldict(case_path: Path, variant: str, config: BenchmarkConfig):
    """Modify controlDict for benchmarking."""
    controldict = case_path / "system" / "controlDict"
    if not controldict.exists():
        return

    content = controldict.read_text()

    # For GPU variant, add libs entry for OGL
    if variant == "gpu":
        if 'libOGL.so' not in content:
            # Add libs directive before the closing comment or at end
            libs_entry = '\nlibs ("libOGL.so");\n'

            # Insert after the FoamFile block
            foamfile_end = content.find('// * * *')
            if foamfile_end > 0:
                # Find the next newline after the comment line
                next_nl = content.find('\n', foamfile_end)
                if next_nl > 0:
                    content = (
                        content[:next_nl + 1]
                        + libs_entry
                        + content[next_nl + 1:]
                    )
            else:
                content += libs_entry

    # Limit timesteps if configured
    if config.max_timesteps > 0:
        content = _limit_timesteps(content, config.max_timesteps)

    # Disable writing intermediate results to save disk space
    content = _minimize_output(content)

    controldict.write_text(content)


def _limit_timesteps(content: str, max_steps: int) -> str:
    """
    Limit the number of timesteps.
    Parse deltaT, set endTime = startTime + max_steps * deltaT.
    """
    # Extract deltaT
    dt_match = re.search(r'deltaT\s+([\d.eE+-]+)\s*;', content)
    start_match = re.search(r'startTime\s+([\d.eE+-]+)\s*;', content)

    if dt_match and start_match:
        try:
            dt = float(dt_match.group(1))
            start = float(start_match.group(1))
            new_end = start + max_steps * dt

            # Replace endTime
            content = re.sub(
                r'endTime\s+[\d.eE+-]+\s*;',
                f'endTime         {new_end};',
                content
            )
        except ValueError:
            pass

    return content


def _minimize_output(content: str) -> str:
    """Reduce file output to save space during benchmarking."""
    # Set purgeWrite to keep only last 2 timesteps
    if re.search(r'purgeWrite\s+\d+\s*;', content):
        content = re.sub(
            r'purgeWrite\s+\d+\s*;',
            'purgeWrite      2;',
            content
        )

    # Use binary format for speed
    content = re.sub(
        r'writeFormat\s+\w+\s*;',
        'writeFormat     binary;',
        content
    )

    return content


def create_allrun_benchmark(case_path: Path, variant: str) -> Path:
    """
    Create a benchmark-specific Allrun script that:
    1. Runs blockMesh (if needed)
    2. Runs the solver
    3. Outputs timing info
    """
    script_path = case_path / "Allrun.benchmark"

    # Check what mesh tools are needed
    has_blockmesh = (case_path / "system" / "blockMeshDict").exists()
    has_snappy = (case_path / "system" / "snappyHexMeshDict").exists()
    has_toposet = (case_path / "system" / "topoSetDict").exists()

    # Determine the solver command
    controldict = case_path / "system" / "controlDict"
    solver_cmd = "foamRun"
    if controldict.exists():
        cdict_content = controldict.read_text()
        # Check for legacy application keyword
        app_match = re.search(r'application\s+(\w+)\s*;', cdict_content)
        if app_match:
            solver_cmd = app_match.group(1)

    lines = [
        '#!/bin/sh',
        'cd "${0%/*}" || exit 1',
        '. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions',
        '',
        '# Mesh generation',
    ]

    if has_blockmesh:
        lines.append('runApplication blockMesh')

    if has_toposet:
        lines.append('runApplication topoSet')

    if has_snappy:
        lines.append('runApplication snappyHexMesh -overwrite')

    lines.extend([
        '',
        '# Run solver',
        f'runApplication {solver_cmd}',
        '',
    ])

    script_content = '\n'.join(lines) + '\n'
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    return script_path
