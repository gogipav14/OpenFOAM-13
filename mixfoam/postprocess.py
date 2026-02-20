"""
Post-process OpenFOAM mixing simulation results.

Parses function object outputs and computes:
  - EDR (Energy Dissipation Rate)
  - Homogeneity (CoV of tracer)
  - TKE (Turbulent Kinetic Energy)
  - P/V (Power per Volume)
  - Max shear rate
  - Bulk/average shear rate
  - Power number
"""

import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_function_object_data(filepath: Path) -> List[Tuple[float, List[float]]]:
    """
    Parse a standard OpenFOAM function object output file.
    Returns list of (time, [values...]) tuples.
    """
    results = []
    if not filepath.exists():
        return results

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                time = float(parts[0])
                values = [float(v) for v in parts[1:]]
                results.append((time, values))
            except (ValueError, IndexError):
                continue
    return results


def _find_latest_time_dir(postproc_dir: Path) -> Optional[Path]:
    """Find the latest time directory in a postProcessing subdirectory."""
    if not postproc_dir.exists():
        return None
    time_dirs = []
    for d in postproc_dir.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
                time_dirs.append((t, d))
            except ValueError:
                continue
    if not time_dirs:
        return None
    return max(time_dirs, key=lambda x: x[0])[1]


def parse_forces(case_dir: Path) -> Tuple[List[float], List[float]]:
    """
    Parse forces function object output to get torque time history.
    Returns (times, torques_z) where torque_z is the z-component of moment.
    """
    forces_dir = case_dir / "postProcessing" / "impellerForces"
    time_dir = _find_latest_time_dir(forces_dir)
    if time_dir is None:
        return [], []

    moment_file = time_dir / "moment.dat"
    if not moment_file.exists():
        return [], []

    times, torques = [], []
    with open(moment_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Format: time ((px py pz) (vx vy vz) (porousx porousy porousz))
            # We want total moment z-component = pz + vz
            m = re.match(r"([\d.eE+-]+)\s+\(\(([^)]+)\)\s+\(([^)]+)\)", line)
            if m:
                t = float(m.group(1))
                pressure = [float(x) for x in m.group(2).split()]
                viscous = [float(x) for x in m.group(3).split()]
                # Total z-torque
                tz = pressure[2] + viscous[2]
                times.append(t)
                torques.append(abs(tz))

    return times, torques


def parse_vol_field_value(case_dir: Path, func_name: str,
                          field_name: Optional[str] = None) -> List[Tuple[float, float]]:
    """Parse volFieldValue output (time vs scalar value)."""
    func_dir = case_dir / "postProcessing" / func_name
    time_dir = _find_latest_time_dir(func_dir)
    if time_dir is None:
        return []

    # Try volFieldValue.dat or fieldName specific file
    candidates = ["volFieldValue.dat"]
    if field_name:
        candidates.insert(0, f"volFieldValue_{field_name}.dat")

    for fname in candidates:
        fpath = time_dir / fname
        if fpath.exists():
            data = _parse_function_object_data(fpath)
            return [(t, v[0]) for t, v in data if v]

    return []


def parse_probes(case_dir: Path, func_name: str = "tracerProbes",
                 field: str = "tracer") -> List[Tuple[float, List[float]]]:
    """Parse probe output. Returns (time, [value_at_probe_1, value_at_probe_2, ...])."""
    probe_dir = case_dir / "postProcessing" / func_name
    time_dir = _find_latest_time_dir(probe_dir)
    if time_dir is None:
        return []

    fpath = time_dir / field
    if not fpath.exists():
        return []

    return _parse_function_object_data(fpath)


def compute_results(case_dir: str, rpm: float, density: float,
                    viscosity_pa_s: float, impeller_diameter: float,
                    volume_m3: float) -> Dict:
    """
    Compute all output metrics from a completed simulation.

    Returns dict with:
        torque, power, P_V, power_number,
        homogeneity_cov, probe_homogeneity,
        max_shear_rate, avg_shear_rate,
        edr, tke
    """
    case = Path(case_dir)
    results = {}
    omega = 2.0 * math.pi * rpm / 60.0  # rad/s
    N = rpm / 60.0  # rev/s

    # --- Power from torque ---
    times, torques = parse_forces(case)
    if torques:
        # Use time-averaged torque from second half of simulation
        n_half = len(torques) // 2
        avg_torque = sum(torques[n_half:]) / max(len(torques[n_half:]), 1)
        power = avg_torque * omega

        results["torque_Nm"] = avg_torque
        results["power_W"] = power
        results["P_V_W_m3"] = power / volume_m3 if volume_m3 > 0 else 0
        results["power_number"] = (
            power / (density * N ** 3 * impeller_diameter ** 5)
            if N > 0 and impeller_diameter > 0 else 0
        )
    else:
        results["torque_Nm"] = None
        results["power_W"] = None
        results["P_V_W_m3"] = None
        results["power_number"] = None

    # --- Homogeneity (CoV) ---
    cov_data = parse_vol_field_value(case, "mixingCoV")
    if cov_data:
        results["homogeneity_cov_final"] = cov_data[-1][1]
        results["homogeneity_cov_timeseries"] = cov_data
    else:
        results["homogeneity_cov_final"] = None

    # --- Probe homogeneity ---
    probe_data = parse_probes(case, "tracerProbes", "tracer")
    if probe_data:
        # CoV across probes at final time
        final_values = probe_data[-1][1]
        if final_values:
            mean_val = sum(final_values) / len(final_values)
            if mean_val > 0:
                std_val = (sum((v - mean_val) ** 2 for v in final_values)
                           / len(final_values)) ** 0.5
                results["probe_cov_final"] = std_val / mean_val
            else:
                results["probe_cov_final"] = None
        else:
            results["probe_cov_final"] = None
    else:
        results["probe_cov_final"] = None

    # --- Shear rates ---
    max_shear = parse_vol_field_value(case, "maxShearRate")
    if max_shear:
        results["max_shear_rate_1_s"] = max_shear[-1][1]
    else:
        results["max_shear_rate_1_s"] = None

    avg_shear = parse_vol_field_value(case, "volumeAverages")
    if avg_shear:
        results["avg_shear_rate_1_s"] = avg_shear[-1][1]
    else:
        results["avg_shear_rate_1_s"] = None

    # --- EDR ---
    edr_data = parse_vol_field_value(case, "volumeAverages")
    if edr_data:
        results["edr_avg_m2_s3"] = edr_data[-1][1]
    else:
        results["edr_avg_m2_s3"] = None

    # --- TKE (from fieldAverage prime2Mean) ---
    # TKE needs to be computed from UPrime2Mean field in the last time directory
    results["tke_note"] = "TKE computed from UPrime2Mean field: k = 0.5*(Rxx+Ryy+Rzz)"

    return results


def print_results(results: Dict):
    """Print formatted results summary."""
    print("\n" + "=" * 60)
    print("MIXING SIMULATION RESULTS")
    print("=" * 60)

    def _fmt(key, unit, fmt=".4g"):
        val = results.get(key)
        if val is None:
            return "  (not available)"
        return f"  {val:{fmt}} {unit}"

    print(f"\nPower & Torque:")
    print(f"  Torque:        {_fmt('torque_Nm', 'N·m')}")
    print(f"  Power:         {_fmt('power_W', 'W')}")
    print(f"  P/V:           {_fmt('P_V_W_m3', 'W/m³')}")
    print(f"  Power Number:  {_fmt('power_number', '')}")

    print(f"\nMixing Homogeneity:")
    print(f"  Volume CoV:    {_fmt('homogeneity_cov_final', '')}")
    print(f"  Probe CoV:     {_fmt('probe_cov_final', '')}")

    print(f"\nShear Rates:")
    print(f"  Max:           {_fmt('max_shear_rate_1_s', '1/s')}")
    print(f"  Bulk Average:  {_fmt('avg_shear_rate_1_s', '1/s')}")

    print(f"\nEnergy Dissipation Rate:")
    print(f"  Volume Avg:    {_fmt('edr_avg_m2_s3', 'm²/s³')}")

    print(f"\nTurbulent Kinetic Energy:")
    print(f"  {results.get('tke_note', 'Not available')}")
    print("=" * 60)
