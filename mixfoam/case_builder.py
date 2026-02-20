"""
Generate a complete OpenFOAM case directory for a mixing simulation.

Orchestrates:
  1. STL extraction and scaling
  2. Mesh template generation (blockMeshDict + snappyHexMeshDict)
  3. Solver and scheme configuration
  4. Boundary condition generation
  5. Post-processing function object setup
"""

import math
import os
from pathlib import Path
from typing import Optional

from . import geometry
from . import stl_extract
from .reactor_db import ReactorConfig
from .templates import (
    blockMeshDict_cylindrical,
    blockMeshDict_rectangular,
    snappyHexMeshDict,
    dynamicMeshDict,
    controlDict,
    fvSolution,
    fvSchemes,
    physicalProperties,
    momentumTransport,
    setFieldsDict,
    decomposeParDict,
    createNonConformalCouplesDict,
    boundary_conditions,
)


def _write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _compute_probe_locations(config: ReactorConfig, liquid_level: float) -> list:
    """Compute TMB probe locations based on tank geometry."""
    tank = config.tank
    probes = []

    if tank.type == "Cylindrical":
        r_probe = tank.diameter / 2.0 * 0.7  # 70% of tank radius
        h_mid = liquid_level * 0.5
        h_low = liquid_level * 0.15
        h_high = liquid_level * 0.85

        # 4 probes at 90° intervals at mid-height
        for angle_deg in [0, 90, 180, 270]:
            angle = math.radians(angle_deg)
            x = r_probe * math.cos(angle)
            y = r_probe * math.sin(angle)
            probes.append((x, y, h_mid))

        # 2 probes near top and bottom at center
        probes.append((0, 0, h_low))
        probes.append((0, 0, h_high))
    else:
        # Rectangular
        lx = tank.length * 0.35
        ly = tank.width * 0.35
        h_mid = liquid_level * 0.5
        h_low = liquid_level * 0.15
        h_high = liquid_level * 0.85

        probes.extend([
            (lx, ly, h_mid),
            (-lx, ly, h_mid),
            (-lx, -ly, h_mid),
            (lx, -ly, h_mid),
            (0, 0, h_low),
            (0, 0, h_high),
        ])

    return probes


def _compute_end_time(config: ReactorConfig, rpm: float) -> float:
    """Estimate simulation end time based on tank turnover."""
    # Aim for ~20 impeller revolutions for initial transient + ~30 for statistics
    if rpm <= 0:
        return 10.0
    rev_per_sec = rpm / 60.0
    return max(50.0 / rev_per_sec, 5.0)


def _compute_delta_t(config: ReactorConfig, rpm: float, target_cell_size: float) -> float:
    """Estimate time step for CFL ≈ 0.5."""
    if rpm <= 0:
        return 1e-3
    # Tip speed
    v_tip = math.pi * config.impeller.diameter * rpm / 60.0
    # CFL = v * dt / dx → dt = CFL * dx / v
    dt = 0.5 * target_cell_size / v_tip
    # Round to nice number
    exp = math.floor(math.log10(dt))
    return round(dt, -exp + 1)


def build_case(
    config: ReactorConfig,
    mdata_path: str,
    output_dir: str,
    volume_liters: float,
    rpm: float,
    density: float,
    viscosity_pa_s: float,
    turbulence: str = "LES",
    use_gpu: bool = True,
    target_cell_size: float = 0.003,
    n_procs: int = 4,
):
    """
    Build a complete OpenFOAM case directory.

    Args:
        config: ReactorConfig from the database
        mdata_path: Path to "MixIT Reactors.mdata"
        output_dir: Where to create the case
        volume_liters: Target fill volume in liters
        rpm: Impeller speed in RPM
        density: Fluid density in kg/m³
        viscosity_pa_s: Dynamic viscosity in Pa·s
        turbulence: "LES" or "RANS"
        use_gpu: Whether to use OGL GPU solver
        target_cell_size: Target background cell size in meters
        n_procs: Number of processors for decomposition
    """
    case = Path(output_dir)
    nu = viscosity_pa_s / density  # kinematic viscosity m²/s
    volume_m3 = volume_liters / 1000.0
    tank = config.tank
    imp = config.impeller

    # --- Compute liquid level ---
    liquid_level = geometry.compute_liquid_level(
        volume_m3, tank.type, tank.diameter, tank.length, tank.width,
        tank.straight_side, tank.bottom_style, tank.bottom_depth,
    )

    # --- Compute mesh parameters ---
    mesh_params = geometry.compute_mesh_parameters(
        tank.type, tank.diameter, tank.length, tank.width,
        liquid_level, tank.bottom_depth,
        imp.clearance, imp.diameter, target_cell_size,
    )

    cell_count = geometry.estimate_cell_count(mesh_params, tank.type)

    # --- Print summary ---
    print(f"Case: {config.display_name}")
    print(f"  Tank: {tank.type} D={tank.diameter:.4f}m Bottom={tank.bottom_style}")
    print(f"  Impeller: {imp.type} D={imp.diameter:.4f}m "
          f"C={imp.clearance:.4f}m {imp.mounting}-mounted")
    if abs(imp.off_center) > 0.001 or abs(imp.off_center_y) > 0.001:
        print(f"  Off-center: ({imp.off_center:.4f}, {imp.off_center_y:.4f})m")
    print(f"  Volume: {volume_liters:.1f}L → Level: {liquid_level:.4f}m")
    print(f"  RPM: {rpm:.0f}  ν={nu:.2e} m²/s  ρ={density:.0f} kg/m³")
    print(f"  Estimated cells: ~{cell_count:,}")
    print(f"  Impeller zone: {mesh_params['zone_bottom']:.4f} - {mesh_params['zone_top']:.4f}m")
    print(f"  Output: {case}")

    # --- Extract STLs ---
    # Find the matching impeller STL name
    impeller_stl_name = None
    for stl in config.available_impeller_stls:
        # Match impeller type to STL filename
        stl_base = stl.replace(".stl", "")
        if imp.type == stl_base or imp.type in stl_base:
            impeller_stl_name = stl
            break
    if impeller_stl_name is None and config.available_impeller_stls:
        # Fallback: use first available
        impeller_stl_name = config.available_impeller_stls[0]

    has_impeller_patch = impeller_stl_name is not None

    if has_impeller_patch:
        # Impeller STLs are unit-normalized — scale by actual diameter
        # Translate to off-center position and clearance height
        imp_translate = (imp.off_center, imp.off_center_y, imp.clearance)
        stl_extract.extract_all_geometry(
            mdata_path, config.reactor_id, impeller_stl_name,
            [b.stl_filename for b in config.baffles if b.stl_filename],
            str(case),
            impeller_diameter=imp.diameter,
            impeller_translate=imp_translate,
        )

    # --- Baffle info ---
    baffle_stl_names = [b.stl_filename for b in config.baffles if b.stl_filename]
    num_baffles = len(baffle_stl_names)

    # --- blockMeshDict ---
    if tank.type == "Cylindrical":
        bmd_content = blockMeshDict_cylindrical.generate({
            "tank_radius": tank.diameter / 2.0,
            "liquid_level": liquid_level,
            "bottom_depth": tank.bottom_depth,
            "zone_bottom": mesh_params["zone_bottom"],
            "zone_top": mesh_params["zone_top"],
            "ri": mesh_params["ri"],
            "n_radial": mesh_params["n_radial"],
            "n_axial_low": mesh_params["n_axial_low"],
            "n_axial_mid": mesh_params["n_axial_mid"],
            "n_axial_top": mesh_params["n_axial_top"],
            "n_circ": mesh_params["n_circ"],
        })
    else:
        bmd_content = blockMeshDict_rectangular.generate({
            "tank_length": tank.length,
            "tank_width": tank.width,
            "liquid_level": liquid_level,
            "zone_bottom": mesh_params["zone_bottom"],
            "zone_top": mesh_params["zone_top"],
            "n_x": mesh_params["n_radial"],
            "n_y": mesh_params["n_circ"],
            "n_axial_low": mesh_params["n_axial_low"],
            "n_axial_mid": mesh_params["n_axial_mid"],
            "n_axial_top": mesh_params["n_axial_top"],
        })
    _write_file(case / "system" / "blockMeshDict", bmd_content)

    # --- snappyHexMeshDict ---
    if has_impeller_patch:
        shm_content = snappyHexMeshDict.generate({
            "has_impeller": True,
            "has_baffle": num_baffles > 0,
            "num_baffles": num_baffles,
            "refine_level_impeller": 3,
            "refine_level_baffle": 2,
            "location_in_mesh_z": (mesh_params["zone_bottom"] + mesh_params["zone_top"]) / 2.0,
        })
        _write_file(case / "system" / "snappyHexMeshDict", shm_content)

    # --- dynamicMeshDict ---
    axis_z = 1.0 if imp.mounting == "Bottom" else -1.0
    dmd_content = dynamicMeshDict.generate({
        "origin_x": imp.off_center,
        "origin_y": imp.off_center_y,
        "axis_z": axis_z,
        "rpm": rpm,
    })
    _write_file(case / "constant" / "dynamicMeshDict", dmd_content)

    # --- physicalProperties ---
    pp_content = physicalProperties.generate({"nu": nu})
    _write_file(case / "constant" / "physicalProperties", pp_content)

    # --- momentumTransport ---
    mt_content = momentumTransport.generate({"turbulence_model": turbulence})
    _write_file(case / "constant" / "momentumTransport", mt_content)

    # --- Probe locations ---
    probes = _compute_probe_locations(config, liquid_level)

    # --- controlDict ---
    end_time = _compute_end_time(config, rpm)
    delta_t = _compute_delta_t(config, rpm, target_cell_size)
    write_interval = max(end_time / 50.0, delta_t * 100)

    cd_content = controlDict.generate({
        "end_time": end_time,
        "delta_t": delta_t,
        "write_interval": write_interval,
        "max_co": 0.5,
        "nu": nu,
        "density": density,
        "impeller_offset_x": imp.off_center,
        "impeller_offset_y": imp.off_center_y,
        "impeller_clearance": imp.clearance,
        "use_gpu": use_gpu,
        "probe_locations": probes,
        "has_impeller_patch": has_impeller_patch,
    })
    _write_file(case / "system" / "controlDict", cd_content)

    # --- fvSolution ---
    fvs_content = fvSolution.generate({"use_gpu": use_gpu})
    _write_file(case / "system" / "fvSolution", fvs_content)

    # --- fvSchemes ---
    fvsc_content = fvSchemes.generate({"turbulence_model": turbulence})
    _write_file(case / "system" / "fvSchemes", fvsc_content)

    # --- setFieldsDict ---
    sfd_params = {
        "tank_type": tank.type,
        "liquid_level": liquid_level,
        "tracer_layer_fraction": 0.1,
    }
    if tank.type == "Cylindrical":
        sfd_params["tank_radius"] = tank.diameter / 2.0
    else:
        sfd_params["tank_length"] = tank.length
        sfd_params["tank_width"] = tank.width
    sfd_content = setFieldsDict.generate(sfd_params)
    _write_file(case / "system" / "setFieldsDict", sfd_content)

    # --- decomposeParDict ---
    dpd_content = decomposeParDict.generate({"n_procs": n_procs})
    _write_file(case / "system" / "decomposeParDict", dpd_content)

    # --- createNonConformalCouplesDict ---
    ncc_content = createNonConformalCouplesDict.generate({})
    _write_file(case / "system" / "createNonConformalCouplesDict", ncc_content)

    # --- topoSetDict (empty - zones defined in blockMesh) ---
    _write_file(case / "system" / "topoSetDict", _toposet_dict())

    # --- Boundary conditions (0/ directory) ---
    bc_files = boundary_conditions.generate_all({
        "turbulence_model": turbulence,
        "has_impeller_patch": has_impeller_patch,
    })
    for filename, content in bc_files.items():
        _write_file(case / "0" / filename, content)

    # --- Allrun script ---
    _write_file(case / "Allrun", _allrun_script(has_impeller_patch, use_gpu))
    os.chmod(case / "Allrun", 0o755)

    # --- Allclean script ---
    _write_file(case / "Allclean", _allclean_script())
    os.chmod(case / "Allclean", 0o755)

    # --- Write case metadata ---
    _write_file(case / "case_info.txt", _case_info(
        config, volume_liters, rpm, density, viscosity_pa_s, nu,
        liquid_level, cell_count, turbulence, use_gpu,
    ))

    print(f"\nCase generated successfully at: {case}")
    print(f"  Run with: cd {case} && ./Allrun")
    return str(case)


def _toposet_dict() -> str:
    return """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Impeller cellZone is defined in blockMesh via block naming.
actions ();

// ************************************************************************* //
"""


def _allrun_script(has_impeller_patch: bool, use_gpu: bool) -> str:
    snappy_line = ""
    if has_impeller_patch:
        snappy_line = "runApplication snappyHexMesh -overwrite\n"

    gpu_check = ""
    if use_gpu:
        gpu_check = """
# Check for OGL library
if [ ! -f "$FOAM_USER_LIBBIN/libOGL.so" ]; then
    echo "Warning: libOGL.so not found. GPU acceleration disabled."
    echo "Build OGL: cd modules/OGL && ./Allwmake"
fi
"""

    return f"""#!/bin/sh
cd "${{0%/*}}" || exit
. ${{WM_PROJECT_DIR:?}}/bin/tools/RunFunctions
{gpu_check}
runApplication blockMesh
{snappy_line}runApplication createNonConformalCouples
runApplication setFields
runApplication $(getApplication)

# For parallel: uncomment below
# runApplication decomposePar
# runParallel $(getApplication)
# runApplication reconstructPar
"""


def _allclean_script() -> str:
    return """#!/bin/sh
cd "${0%/*}" || exit
. ${WM_PROJECT_DIR:?}/bin/tools/CleanFunctions

cleanCase
rm -rf constant/geometry/*.stl 2>/dev/null
rm -rf constant/polyMesh 2>/dev/null
"""


def _case_info(config, volume_liters, rpm, density, viscosity, nu,
               liquid_level, cell_count, turbulence, use_gpu) -> str:
    imp = config.impeller
    tank = config.tank
    return f"""MixFOAM Case Information
========================
Reactor: {config.reactor_id} / {config.config_name}
Tank: {tank.type} D={tank.diameter:.4f}m Bottom={tank.bottom_style}
Impeller: {imp.type} D={imp.diameter:.4f}m C={imp.clearance:.4f}m {imp.mounting}-mounted
Off-center: ({imp.off_center:.4f}, {imp.off_center_y:.4f})m

Inputs:
  Volume: {volume_liters:.1f} L
  RPM: {rpm:.0f}
  Density: {density:.1f} kg/m³
  Dynamic viscosity: {viscosity:.2e} Pa·s
  Kinematic viscosity: {nu:.2e} m²/s

Computed:
  Liquid level: {liquid_level:.4f} m
  Estimated cells: ~{cell_count:,}
  Tip speed: {math.pi * imp.diameter * rpm / 60:.2f} m/s
  Re_imp: {density * (rpm/60) * imp.diameter**2 / viscosity:.0f}

Settings:
  Turbulence: {turbulence}
  GPU: {use_gpu}
"""
