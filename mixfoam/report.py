"""
Generate a self-contained HTML report from a completed OpenFOAM mixing simulation.

Reads case metadata from case_info.txt, numeric results from postProcessing/,
and VTK slice data to produce a single HTML file with embedded plots and metrics.

Usage:
    from mixfoam.report import generate_report
    generate_report("/path/to/case")
"""

import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_case_info(case_dir: Path) -> Dict[str, Any]:
    """
    Parse case_info.txt into a structured dict.

    Expected format (produced by case_builder._case_info):
        Reactor: Mobius_MIX / 100L
        Tank: Cylindrical D=0.4500m Bottom=Torispherical
        Impeller: HE3 D=0.2250m C=0.1125m top-mounted
        ...
        Inputs:
          Volume: 50.0 L
          RPM: 200
          Density: 1000.0 kg/m³
          Dynamic viscosity: 1.00e-03 Pa·s
          Kinematic viscosity: 1.00e-06 m²/s
        Computed:
          Liquid level: 0.3142 m
          Estimated cells: ~450,000
          Tip speed: 2.36 m/s
          Re_imp: 101250
        Settings:
          Turbulence: LES
          GPU: True
    """
    info: Dict[str, Any] = {}
    info_path = case_dir / "case_info.txt"
    raw_text = ""

    if not info_path.exists():
        logger.warning("case_info.txt not found in %s", case_dir)
        return info

    raw_text = info_path.read_text(encoding="utf-8", errors="replace")
    info["_raw"] = raw_text

    # --- Reactor line ---
    m = re.search(r"Reactor:\s*(.+?)\s*/\s*(.+)", raw_text)
    if m:
        info["reactor_name"] = m.group(1).strip()
        info["config_name"] = m.group(2).strip()

    # --- Tank line ---
    m = re.search(r"Tank:\s*(.+)", raw_text)
    if m:
        info["tank_desc"] = m.group(1).strip()

    # --- Impeller line ---
    m = re.search(r"Impeller:\s*(.+)", raw_text)
    if m:
        info["impeller_desc"] = m.group(1).strip()

    # --- Impeller diameter from impeller line ---
    m = re.search(r"Impeller:.*?D=([\d.]+)", raw_text)
    if m:
        info["impeller_diameter"] = float(m.group(1))

    # --- Inputs section ---
    m = re.search(r"Volume:\s*([\d.]+)\s*L", raw_text)
    if m:
        info["volume_L"] = float(m.group(1))

    m = re.search(r"RPM:\s*([\d.]+)", raw_text)
    if m:
        info["rpm"] = float(m.group(1))

    m = re.search(r"Density:\s*([\d.]+)\s*kg", raw_text)
    if m:
        info["density"] = float(m.group(1))

    m = re.search(r"Dynamic viscosity:\s*([\d.eE+-]+)\s*Pa", raw_text)
    if m:
        info["viscosity"] = float(m.group(1))

    m = re.search(r"Kinematic viscosity:\s*([\d.eE+-]+)\s*m", raw_text)
    if m:
        info["nu"] = float(m.group(1))

    # --- Computed section ---
    m = re.search(r"Liquid level:\s*([\d.]+)\s*m", raw_text)
    if m:
        info["liquid_level"] = float(m.group(1))

    m = re.search(r"Estimated cells:\s*~?([\d,]+)", raw_text)
    if m:
        info["estimated_cells"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Tip speed:\s*([\d.]+)\s*m/s", raw_text)
    if m:
        info["tip_speed"] = float(m.group(1))

    m = re.search(r"Re_imp:\s*([\d.eE+-]+)", raw_text)
    if m:
        info["Re"] = float(m.group(1))

    # --- Settings ---
    m = re.search(r"Turbulence:\s*(\S+)", raw_text)
    if m:
        info["turbulence"] = m.group(1)

    m = re.search(r"GPU:\s*(\S+)", raw_text)
    if m:
        info["gpu"] = m.group(1)

    return info


def _safe_format(value, fmt: str = ".4g", suffix: str = "") -> str:
    """Format a numeric value, returning 'N/A' if None."""
    if value is None:
        return "N/A"
    try:
        return f"{value:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _fig_to_div(fig) -> str:
    """Convert a plotly figure to an HTML <div> string (no full page, no JS)."""
    if fig is None:
        return ""
    try:
        return fig.to_html(full_html=False, include_plotlyjs=False)
    except Exception as exc:
        logger.warning("Failed to convert figure to HTML: %s", exc)
        return f'<div class="plot-error">Plot generation failed: {exc}</div>'


def _get_plotly_js() -> str:
    """
    Get the plotly.js source code for embedding.

    Tries offline bundle first, falls back to a CDN script tag placeholder.
    """
    try:
        import plotly.offline
        return plotly.offline.get_plotlyjs()
    except Exception:
        logger.warning("Could not load plotly.js offline bundle; using CDN fallback.")
        return ""


def _get_plotly_cdn_fallback() -> str:
    """Return a CDN script tag for plotly.js as a fallback."""
    return '<script src="https://cdn.plot.ly/plotly-2.35.0.min.js" charset="utf-8"></script>'


# ---------------------------------------------------------------------------
# Visualization wrappers — each returns a plotly Figure or None
# ---------------------------------------------------------------------------

def _make_velocity_horizontal(case_dir: Path):
    """Generate horizontal velocity slice colormap."""
    try:
        from mixfoam.visualize import plot_velocity_magnitude
        vtk_dir = str(case_dir / "postProcessing" / "horizontalSlice")
        return plot_velocity_magnitude(vtk_dir, title="Velocity — horizontal slice")
    except Exception as exc:
        logger.warning("Horizontal velocity plot failed: %s", exc)
        return None


def _make_velocity_vertical(case_dir: Path):
    """Generate vertical velocity slice colormap."""
    try:
        from mixfoam.visualize import plot_velocity_magnitude
        vtk_dir = str(case_dir / "postProcessing" / "verticalSlice")
        return plot_velocity_magnitude(vtk_dir, title="Velocity — vertical slice")
    except Exception as exc:
        logger.warning("Vertical velocity plot failed: %s", exc)
        return None


def _make_tracer_animation(case_dir: Path):
    """Generate tracer mixing evolution animation/slider."""
    try:
        from mixfoam.visualize import create_tracer_animation
        vtk_dir = str(case_dir / "postProcessing" / "horizontalSlice")
        return create_tracer_animation(vtk_dir)
    except Exception as exc:
        logger.warning("Tracer animation failed: %s", exc)
        return None


def _make_shear_rate_field(case_dir: Path):
    """Generate shear rate field colormap."""
    try:
        from mixfoam.visualize import plot_shear_rate_field
        vtk_dir = str(case_dir / "postProcessing" / "horizontalSlice")
        return plot_shear_rate_field(vtk_dir)
    except Exception as exc:
        logger.warning("Shear rate field plot failed: %s", exc)
        return None


def _make_shear_histogram(case_dir: Path):
    """Generate strain rate histogram."""
    try:
        from mixfoam.visualize import plot_strain_rate_histogram
        vtk_dir = str(case_dir / "postProcessing" / "horizontalSlice")
        return plot_strain_rate_histogram(vtk_dir)
    except Exception as exc:
        logger.warning("Shear histogram failed: %s", exc)
        return None


def _make_cov_plot(case_dir: Path):
    """Generate CoV vs time plot."""
    try:
        from mixfoam.visualize import plot_mixing_cov
        postproc_dir = str(case_dir / "postProcessing")
        return plot_mixing_cov(postproc_dir)
    except Exception as exc:
        logger.warning("CoV plot failed: %s", exc)
        return None


def _make_torque_plot(case_dir: Path):
    """Generate torque vs time plot."""
    try:
        from mixfoam.visualize import plot_torque_history
        postproc_dir = str(case_dir / "postProcessing")
        return plot_torque_history(postproc_dir)
    except Exception as exc:
        logger.warning("Torque plot failed: %s", exc)
        return None


def _make_shear_history_plot(case_dir: Path):
    """Generate shear rate vs time convergence plot."""
    try:
        from mixfoam.visualize import plot_shear_rate_history
        postproc_dir = str(case_dir / "postProcessing")
        return plot_shear_rate_history(postproc_dir)
    except Exception as exc:
        logger.warning("Shear history plot failed: %s", exc)
        return None


def _make_probe_plot(case_dir: Path):
    """Generate probe data plot (tracer at TMB locations)."""
    try:
        from mixfoam.visualize import plot_probe_data
        postproc_dir = str(case_dir / "postProcessing")
        return plot_probe_data(postproc_dir)
    except Exception as exc:
        logger.warning("Probe plot failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------

def generate_report(case_dir: str, output_path: str = None) -> str:
    """
    Generate a self-contained HTML report for a MixFOAM simulation.

    Args:
        case_dir: Path to the completed OpenFOAM case directory.
        output_path: Where to write the HTML file (default: case_dir/report.html).

    Returns:
        Path to the generated HTML file.

    Report sections:
    1. Header with reactor name, config, logo placeholder
    2. Operating Conditions table (volume, RPM, density, viscosity, Re)
    3. Summary Results table (7 key metrics: EDR, P/V, CoV, TKE, shear rates, Np)
    4. Velocity Field — horizontal + vertical slice colormaps
    5. Tracer/Mixing — animated slider showing mixing evolution over time
    6. Shear Rate Distribution — histogram + field colormap
    7. Time Series — CoV, torque, shear rate convergence plots
    8. Probe Data — tracer at TMB locations
    9. Simulation Details footer (mesh count, solver, turbulence model, runtime)
    """
    import jinja2

    case_path = Path(case_dir).resolve()
    if not case_path.is_dir():
        raise FileNotFoundError(f"Case directory not found: {case_path}")

    if output_path is None:
        output_path = str(case_path / "report.html")

    logger.info("Generating MixFOAM report for %s", case_path)

    # ------------------------------------------------------------------
    # 1. Read case metadata
    # ------------------------------------------------------------------
    case_info = _read_case_info(case_path)
    reactor_name = case_info.get("reactor_name", "Unknown Reactor")
    config_name = case_info.get("config_name", "")
    title = f"MixFOAM Report: {reactor_name}"
    if config_name:
        title += f" / {config_name}"

    # ------------------------------------------------------------------
    # 2. Operating conditions
    # ------------------------------------------------------------------
    operating_conditions = {
        "volume_L": case_info.get("volume_L"),
        "rpm": case_info.get("rpm"),
        "density": case_info.get("density"),
        "viscosity": case_info.get("viscosity"),
        "nu": case_info.get("nu"),
        "Re": case_info.get("Re"),
        "tip_speed": case_info.get("tip_speed"),
    }

    # ------------------------------------------------------------------
    # 3. Compute numeric results
    # ------------------------------------------------------------------
    results: Dict[str, Any] = {
        "edr": None,
        "p_v": None,
        "power_number": None,
        "max_shear": None,
        "avg_shear": None,
        "cov": None,
        "tke": None,
    }

    try:
        from mixfoam.postprocess import compute_results as _compute
        rpm = case_info.get("rpm", 0.0)
        density = case_info.get("density", 1000.0)
        viscosity = case_info.get("viscosity", 0.001)
        imp_d = case_info.get("impeller_diameter", 0.0)
        vol_L = case_info.get("volume_L", 0.0)
        vol_m3 = vol_L / 1000.0 if vol_L else 0.0

        if rpm and imp_d:
            raw = _compute(
                case_dir=str(case_path),
                rpm=rpm,
                density=density,
                viscosity_pa_s=viscosity,
                impeller_diameter=imp_d,
                volume_m3=vol_m3,
            )
            results["edr"] = raw.get("edr_avg_m2_s3")
            results["p_v"] = raw.get("P_V_W_m3")
            results["power_number"] = raw.get("power_number")
            results["max_shear"] = raw.get("max_shear_rate_1_s")
            results["avg_shear"] = raw.get("avg_shear_rate_1_s")
            results["cov"] = raw.get("homogeneity_cov_final")
            results["tke"] = raw.get("tke")
        else:
            logger.warning(
                "Insufficient metadata (rpm=%.1f, imp_d=%.4f) to compute results; "
                "metrics will show N/A.",
                rpm or 0, imp_d or 0,
            )
    except Exception as exc:
        logger.warning("compute_results() failed: %s — metrics will show N/A.", exc)

    # ------------------------------------------------------------------
    # 4. Generate plots via mixfoam.visualize
    # ------------------------------------------------------------------
    logger.info("Generating plots...")

    velocity_h_fig = _make_velocity_horizontal(case_path)
    velocity_v_fig = _make_velocity_vertical(case_path)
    tracer_fig = _make_tracer_animation(case_path)
    shear_field_fig = _make_shear_rate_field(case_path)
    histogram_fig = _make_shear_histogram(case_path)
    cov_fig = _make_cov_plot(case_path)
    torque_fig = _make_torque_plot(case_path)
    shear_hist_fig = _make_shear_history_plot(case_path)
    probe_fig = _make_probe_plot(case_path)

    # Convert plotly figures to HTML div strings
    velocity_h_plot = _fig_to_div(velocity_h_fig)
    velocity_v_plot = _fig_to_div(velocity_v_fig)
    tracer_animation = _fig_to_div(tracer_fig)
    shear_rate_plot = _fig_to_div(shear_field_fig)
    histogram_plot = _fig_to_div(histogram_fig)
    cov_plot = _fig_to_div(cov_fig)
    torque_plot = _fig_to_div(torque_fig)
    shear_history_plot = _fig_to_div(shear_hist_fig)
    probe_plot = _fig_to_div(probe_fig)

    # ------------------------------------------------------------------
    # 5. Plotly.js for embedding
    # ------------------------------------------------------------------
    plotly_js = _get_plotly_js()
    plotly_cdn_fallback = _get_plotly_cdn_fallback() if not plotly_js else ""

    # ------------------------------------------------------------------
    # 6. Render the Jinja2 template
    # ------------------------------------------------------------------
    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = template.render(
        title=title,
        reactor_name=reactor_name,
        config_name=config_name,
        generation_time=generation_time,
        operating_conditions=operating_conditions,
        results=results,
        plotly_js=plotly_js,
        plotly_cdn_fallback=plotly_cdn_fallback,
        velocity_h_plot=velocity_h_plot,
        velocity_v_plot=velocity_v_plot,
        tracer_animation=tracer_animation,
        shear_rate_plot=shear_rate_plot,
        histogram_plot=histogram_plot,
        cov_plot=cov_plot,
        torque_plot=torque_plot,
        shear_history_plot=shear_history_plot,
        probe_plot=probe_plot,
        case_info_text=case_info.get("_raw", ""),
        # Simulation details for footer
        estimated_cells=case_info.get("estimated_cells"),
        turbulence=case_info.get("turbulence", "N/A"),
        gpu=case_info.get("gpu", "N/A"),
        tank_desc=case_info.get("tank_desc", "N/A"),
        impeller_desc=case_info.get("impeller_desc", "N/A"),
    )

    # ------------------------------------------------------------------
    # 7. Write output
    # ------------------------------------------------------------------
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_content, encoding="utf-8")

    logger.info("Report written to %s", output)
    return str(output)
