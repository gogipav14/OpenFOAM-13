#!/usr/bin/env python3
"""
MixFOAM CLI — Generate and run OpenFOAM mixing simulations from MixIT reactor data.

Usage:
    python -m mixfoam list [--archive PATH]
    python -m mixfoam info <reactor> [config] [--archive PATH]
    python -m mixfoam setup <reactor> <config> [options]
    python -m mixfoam run <case_dir> [--parallel] [--nprocs N]
    python -m mixfoam results <case_dir> [options]
"""

import argparse
import os
import sys
from pathlib import Path

# Default archive location (same directory as the repo)
DEFAULT_ARCHIVE = str(
    Path(__file__).parent.parent.parent / "MixIT Reactors.mdata"
)


def cmd_list(args):
    """List all available reactors and configurations."""
    from .reactor_db import load_reactor_database, print_reactor_summary

    db = load_reactor_database(args.archive)
    print_reactor_summary(db)


def cmd_info(args):
    """Show detailed info for a reactor/configuration."""
    from .reactor_db import load_reactor_database

    db = load_reactor_database(args.archive)
    family = db.get(args.reactor)
    if family is None:
        print(f"Error: Reactor '{args.reactor}' not found.")
        print(f"Available: {', '.join(sorted(db.keys()))}")
        sys.exit(1)

    if args.config:
        configs = [c for c in family.configs if c.config_name == args.config]
        if not configs:
            print(f"Error: Config '{args.config}' not found for {args.reactor}.")
            print(f"Available: {', '.join(c.config_name for c in family.configs)}")
            sys.exit(1)
    else:
        configs = family.configs

    for cfg in configs:
        t = cfg.tank
        imp = cfg.impeller
        op = cfg.operating

        print(f"\n{'='*60}")
        print(f"Reactor: {cfg.reactor_id} / {cfg.config_name}")
        print(f"  Process: {cfg.process_name}  Scale: {cfg.scale}")
        print(f"\nTank:")
        print(f"  Type: {t.type}")
        print(f"  Diameter: {t.diameter:.4f} m")
        if t.type == "Rectangular":
            print(f"  Length: {t.length:.4f} m  Width: {t.width:.4f} m")
        print(f"  Straight side: {t.straight_side:.4f} m")
        print(f"  Bottom: {t.bottom_style} (depth: {t.bottom_depth:.4f} m)")
        print(f"  Head: {t.head_style} (depth: {t.head_depth:.4f} m)")

        print(f"\nImpeller:")
        print(f"  Type: {imp.type}")
        print(f"  Diameter: {imp.diameter:.4f} m")
        print(f"  Clearance: {imp.clearance:.4f} m")
        print(f"  Mounting: {imp.mounting}")
        print(f"  Off-center: ({imp.off_center:.4f}, {imp.off_center_y:.4f}) m")
        print(f"  Blades: {imp.number_of_blades}")

        print(f"\nDefault Operating Conditions:")
        print(f"  Volume: {op.volume_m3*1000:.1f} L")
        print(f"  Liquid level: {op.liquid_level:.4f} m")
        print(f"  RPM: {op.rpm:.0f}")

        print(f"\nBaffles: {len(cfg.baffles)}")
        for i, b in enumerate(cfg.baffles):
            print(f"  [{i+1}] {b.style} W={b.width:.4f}m"
                  f"  STL: {b.stl_filename or 'none'}")

        print(f"\nAvailable impeller STLs:")
        for s in cfg.available_impeller_stls:
            marker = " <-- matches" if imp.type in s.replace(".stl", "") else ""
            print(f"  {s}{marker}")

        print(f"\nAvailable baffle STLs:")
        for s in cfg.available_baffle_stls:
            print(f"  {s}")


def cmd_setup(args):
    """Generate an OpenFOAM case directory."""
    from .reactor_db import load_reactor_database, find_config
    from .case_builder import build_case

    db = load_reactor_database(args.archive)
    config = find_config(db, args.reactor, args.config)
    if config is None:
        print(f"Error: Config '{args.reactor}/{args.config}' not found.")
        sys.exit(1)

    build_case(
        config=config,
        mdata_path=args.archive,
        output_dir=args.output,
        volume_liters=args.volume,
        rpm=args.rpm,
        density=args.density,
        viscosity_pa_s=args.viscosity,
        turbulence=args.turbulence,
        use_gpu=not args.no_gpu,
        target_cell_size=args.cell_size,
        n_procs=args.nprocs,
    )


def cmd_run(args):
    """Run the OpenFOAM simulation."""
    from .runner import run_case

    success = run_case(
        case_dir=args.case_dir,
        parallel=args.parallel,
        n_procs=args.nprocs,
    )
    sys.exit(0 if success else 1)


def cmd_results(args):
    """Post-process and display results."""
    from .postprocess import compute_results, print_results

    results = compute_results(
        case_dir=args.case_dir,
        rpm=args.rpm,
        density=args.density,
        viscosity_pa_s=args.viscosity,
        impeller_diameter=args.impeller_diameter,
        volume_m3=args.volume / 1000.0,
    )
    print_results(results)


def main():
    parser = argparse.ArgumentParser(
        prog="mixfoam",
        description="MixFOAM — MixIT reactor CFD simulation wrapper for OpenFOAM",
    )
    parser.add_argument(
        "--archive", default=DEFAULT_ARCHIVE,
        help="Path to MixIT Reactors.mdata archive",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # --- list ---
    sub_list = subparsers.add_parser("list", help="List all reactors and configs")

    # --- info ---
    sub_info = subparsers.add_parser("info", help="Show reactor details")
    sub_info.add_argument("reactor", help="Reactor family name (e.g. Mobius_MIX)")
    sub_info.add_argument("config", nargs="?", default=None,
                          help="Configuration name (e.g. 100L)")

    # --- setup ---
    sub_setup = subparsers.add_parser("setup", help="Generate OpenFOAM case")
    sub_setup.add_argument("reactor", help="Reactor family name")
    sub_setup.add_argument("config", help="Configuration name")
    sub_setup.add_argument("--volume", type=float, required=True,
                           help="Fill volume in liters")
    sub_setup.add_argument("--rpm", type=float, required=True,
                           help="Impeller speed in RPM")
    sub_setup.add_argument("--density", type=float, default=1000.0,
                           help="Fluid density in kg/m³ (default: 1000)")
    sub_setup.add_argument("--viscosity", type=float, default=0.001,
                           help="Dynamic viscosity in Pa·s (default: 0.001)")
    sub_setup.add_argument("--turbulence", choices=["LES", "RANS"], default="LES",
                           help="Turbulence model (default: LES)")
    sub_setup.add_argument("--no-gpu", action="store_true",
                           help="Disable GPU solver (use CPU PCG)")
    sub_setup.add_argument("--cell-size", type=float, default=0.003,
                           help="Target cell size in meters (default: 0.003)")
    sub_setup.add_argument("--nprocs", type=int, default=4,
                           help="Number of processors for decomposition")
    sub_setup.add_argument("--output", "-o", default="./mixfoam_case",
                           help="Output case directory (default: ./mixfoam_case)")

    # --- run ---
    sub_run = subparsers.add_parser("run", help="Run simulation")
    sub_run.add_argument("case_dir", help="Path to case directory")
    sub_run.add_argument("--parallel", action="store_true",
                         help="Run in parallel (multi-GPU)")
    sub_run.add_argument("--nprocs", type=int, default=4,
                         help="Number of processors")

    # --- results ---
    sub_results = subparsers.add_parser("results", help="Post-process results")
    sub_results.add_argument("case_dir", help="Path to case directory")
    sub_results.add_argument("--rpm", type=float, required=True)
    sub_results.add_argument("--density", type=float, default=1000.0)
    sub_results.add_argument("--viscosity", type=float, default=0.001)
    sub_results.add_argument("--impeller-diameter", type=float, required=True)
    sub_results.add_argument("--volume", type=float, required=True,
                             help="Fill volume in liters")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cmd_map = {
        "list": cmd_list,
        "info": cmd_info,
        "setup": cmd_setup,
        "run": cmd_run,
        "results": cmd_results,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
