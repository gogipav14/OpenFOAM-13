"""
Visualization module for MixFOAM CFD post-processing.

Reads OpenFOAM postProcessing data and generates self-contained plotly
figures for an HTML report.  Handles two data sources:

  1. VTK surface slices from ``postProcessing/horizontalSlice/`` and
     ``postProcessing/verticalSlice/`` (read with pyvista).
  2. Time-series text files from various function-object subdirectories
     (parsed with stdlib + optional numpy).

All figures use a dark "director-grade" theme with large fonts, professional
colormaps, and dark backgrounds.  Missing data or missing dependencies
(pyvista) are handled gracefully with informative placeholder figures.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import plotly.graph_objects as go

# Optional heavy dependencies -- module must load without them.
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import pyvista as pv

    _HAS_PYVISTA = True
except ImportError:  # pragma: no cover
    pv = None  # type: ignore[assignment]
    _HAS_PYVISTA = False

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
DARK_BG = "rgb(17,17,17)"
DARK_PAPER = "rgb(30,30,30)"
GRID_COLOR = "rgb(50,50,50)"
TEXT_COLOR = "rgb(220,220,220)"
FONT_FAMILY = "Arial, sans-serif"

_BASE_LAYOUT = dict(
    plot_bgcolor=DARK_BG,
    paper_bgcolor=DARK_PAPER,
    font=dict(family=FONT_FAMILY, size=14, color=TEXT_COLOR),
    title_font=dict(size=20, color=TEXT_COLOR),
    margin=dict(l=70, r=40, t=60, b=60),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
)


def _base_layout(**overrides) -> dict:
    """Return a copy of the base dark layout with optional overrides."""
    layout = {}
    for key, val in _BASE_LAYOUT.items():
        layout[key] = dict(val) if isinstance(val, dict) else val
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# Placeholder figure helper
# ---------------------------------------------------------------------------

def _placeholder_figure(message: str, title: str = "") -> go.Figure:
    """Return an empty figure that displays *message* as a centred annotation."""
    fig = go.Figure()
    fig.update_layout(
        **_base_layout(
            title=dict(text=title) if title else {},
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text=message,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=18, color="rgb(180,180,180)"),
                    xanchor="center",
                    yanchor="middle",
                ),
            ],
        ),
    )
    return fig


# ===================================================================
# VTK slice reading
# ===================================================================

def get_available_timesteps(vtk_dir: str) -> List[str]:
    """List available timestep directories in a postProcessing VTK directory.

    Parameters
    ----------
    vtk_dir : str
        Path to a postProcessing function-object directory, e.g.
        ``<case>/postProcessing/horizontalSlice``.

    Returns
    -------
    list[str]
        Sorted list of timestep directory names (ascending by float value).
    """
    vtk_path = Path(vtk_dir)
    if not vtk_path.is_dir():
        return []

    timesteps: List[Tuple[float, str]] = []
    for entry in vtk_path.iterdir():
        if entry.is_dir():
            try:
                t = float(entry.name)
                timesteps.append((t, entry.name))
            except ValueError:
                continue

    timesteps.sort(key=lambda x: x[0])
    return [name for _, name in timesteps]


def _resolve_timestep(vtk_dir: str, time_step: Optional[str]) -> Optional[str]:
    """Return an explicit timestep string, defaulting to the latest available."""
    if time_step is not None:
        return time_step
    available = get_available_timesteps(vtk_dir)
    return available[-1] if available else None


def _find_vtk_file(vtk_dir: str, time_step: str) -> Optional[Path]:
    """Locate a VTK file inside *vtk_dir*/*time_step*/.

    OpenFOAM writes surfaces with names like ``zSlice.vtk`` or
    ``ySlice.vtk`` inside the timestep folder.  We return the first
    ``.vtk`` file found.
    """
    ts_path = Path(vtk_dir) / time_step
    if not ts_path.is_dir():
        return None

    vtk_files = sorted(ts_path.glob("*.vtk"))
    if vtk_files:
        return vtk_files[0]

    # Some versions nest one level deeper (e.g. <time>/surfaceName.vtk
    # inside a subdirectory).
    for child in ts_path.iterdir():
        if child.is_dir():
            nested = sorted(child.glob("*.vtk"))
            if nested:
                return nested[0]

    return None


def read_vtk_slice(vtk_dir: str, time_step: Optional[str] = None) -> dict:
    """Read a VTK surface slice from postProcessing.

    Parameters
    ----------
    vtk_dir : str
        Path to the function-object directory, e.g.
        ``<case>/postProcessing/horizontalSlice``.
    time_step : str, optional
        Timestep directory name.  If *None*, the latest available is used.

    Returns
    -------
    dict
        Keys: ``points`` (N x 3 ndarray), and any available field arrays
        (``U``, ``p``, ``tracer``, ``shearRate``, ...) as ndarrays.
        Returns an empty dict if the data cannot be read.
    """
    if not _HAS_PYVISTA:
        return {}

    ts = _resolve_timestep(vtk_dir, time_step)
    if ts is None:
        return {}

    vtk_file = _find_vtk_file(vtk_dir, ts)
    if vtk_file is None:
        return {}

    try:
        mesh = pv.read(str(vtk_file))
    except Exception:
        return {}

    result: dict = {"points": np.asarray(mesh.points)}

    # Collect all available point/cell data arrays.
    for name in mesh.point_data:
        result[name] = np.asarray(mesh.point_data[name])
    for name in mesh.cell_data:
        if name not in result:
            result[name] = np.asarray(mesh.cell_data[name])

    return result


# ===================================================================
# Field colormaps (2-D slice plots)
# ===================================================================

def _scatter_colormap(
    data: dict,
    field_values: "np.ndarray",
    colorscale: str,
    cbar_title: str,
    title: str,
    log_scale: bool = False,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """Build a 2-D scatter-based colormap from slice data."""
    pts = data["points"]
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # Determine whether the slice is horizontal (z ~ const) or vertical.
    z_range = float(z.max() - z.min())
    x_range = float(x.max() - x.min())
    y_range = float(y.max() - y.min())

    if z_range < 0.1 * max(x_range, y_range, 1e-12):
        # Horizontal slice -- plot x vs y.
        plot_x, plot_y = x, y
        xlab, ylab = "x [m]", "y [m]"
    elif y_range < 0.1 * max(x_range, z_range, 1e-12):
        # Vertical slice, constant y -- plot x vs z.
        plot_x, plot_y = x, z
        xlab, ylab = "x [m]", "z [m]"
    else:
        # Vertical slice, constant x -- plot y vs z.
        plot_x, plot_y = y, z
        xlab, ylab = "y [m]", "z [m]"

    vals = field_values.copy().astype(float)
    if log_scale:
        vals = np.where(vals > 0, np.log10(vals), np.nan)
        cbar_title = f"log10({cbar_title})"

    if zmin is not None and zmax is not None:
        pass
    elif log_scale:
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            zmin = float(np.percentile(finite, 1))
            zmax = float(np.percentile(finite, 99))
    else:
        zmin = float(np.nanpercentile(vals, 1))
        zmax = float(np.nanpercentile(vals, 99))

    fig = go.Figure(
        data=go.Scattergl(
            x=plot_x,
            y=plot_y,
            mode="markers",
            marker=dict(
                color=vals,
                colorscale=colorscale,
                cmin=zmin,
                cmax=zmax,
                size=3,
                colorbar=dict(
                    title=dict(text=cbar_title, font=dict(color=TEXT_COLOR)),
                    tickfont=dict(color=TEXT_COLOR),
                ),
            ),
            hovertemplate=(
                f"{xlab}: %{{x:.4f}}<br>{ylab}: %{{y:.4f}}"
                f"<br>{cbar_title}: %{{marker.color:.4g}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title),
            xaxis=dict(title=xlab, scaleanchor="y", scaleratio=1,
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            yaxis=dict(title=ylab,
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        )
    )
    return fig


def plot_velocity_magnitude(
    vtk_dir: str,
    time_step: Optional[str] = None,
    title: str = "Velocity magnitude",
) -> go.Figure:
    """Velocity magnitude colormap from a VTK slice.

    Uses the *Turbo* colormap on a dark background.
    """
    if not _HAS_PYVISTA:
        return _placeholder_figure(
            "pyvista is not installed -- cannot read VTK slices.", title
        )

    data = read_vtk_slice(vtk_dir, time_step)
    if not data:
        return _placeholder_figure("No VTK slice data found.", title)

    if "U" not in data:
        return _placeholder_figure(
            "Velocity field (U) not found in slice data.", title
        )

    u_arr = data["U"]
    if u_arr.ndim == 2 and u_arr.shape[1] >= 3:
        mag = np.linalg.norm(u_arr[:, :3], axis=1)
    else:
        mag = np.abs(u_arr)

    return _scatter_colormap(
        data, mag, colorscale="Turbo", cbar_title="|U| [m/s]", title=title,
    )


def plot_velocity_vectors(
    vtk_dir: str,
    time_step: Optional[str] = None,
    title: str = "Velocity vectors",
) -> go.Figure:
    """Velocity quiver / vector plot on a 2-D slice."""
    if not _HAS_PYVISTA:
        return _placeholder_figure(
            "pyvista is not installed -- cannot read VTK slices.", title
        )

    data = read_vtk_slice(vtk_dir, time_step)
    if not data:
        return _placeholder_figure("No VTK slice data found.", title)

    if "U" not in data:
        return _placeholder_figure(
            "Velocity field (U) not found in slice data.", title
        )

    pts = data["points"]
    u_arr = data["U"]

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    z_range = float(z.max() - z.min())
    x_range = float(x.max() - x.min())
    y_range = float(y.max() - y.min())

    if z_range < 0.1 * max(x_range, y_range, 1e-12):
        plot_x, plot_y = x, y
        uc, vc = u_arr[:, 0], u_arr[:, 1]
        xlab, ylab = "x [m]", "y [m]"
    elif y_range < 0.1 * max(x_range, z_range, 1e-12):
        plot_x, plot_y = x, z
        uc, vc = u_arr[:, 0], u_arr[:, 2]
        xlab, ylab = "x [m]", "z [m]"
    else:
        plot_x, plot_y = y, z
        uc, vc = u_arr[:, 1], u_arr[:, 2]
        xlab, ylab = "y [m]", "z [m]"

    mag = np.sqrt(uc ** 2 + vc ** 2)

    # Subsample for readability -- target ~1500 arrows.
    n_pts = len(plot_x)
    stride = max(1, n_pts // 1500)
    idx = np.arange(0, n_pts, stride)

    # Normalise arrow length for visual clarity.
    max_mag = float(mag[idx].max()) if len(idx) > 0 else 1.0
    if max_mag < 1e-30:
        max_mag = 1.0
    scale = float(max(x_range, y_range)) * 0.03 / max_mag

    fig = go.Figure()

    # Cone / quiver approximation using annotations is heavy; instead use
    # a scatter for the base points coloured by magnitude and overlay
    # line segments for direction.
    fig.add_trace(
        go.Scattergl(
            x=plot_x[idx],
            y=plot_y[idx],
            mode="markers",
            marker=dict(
                color=mag[idx],
                colorscale="Turbo",
                size=3,
                colorbar=dict(
                    title=dict(text="|U| [m/s]", font=dict(color=TEXT_COLOR)),
                    tickfont=dict(color=TEXT_COLOR),
                ),
            ),
            hovertemplate=(
                f"{xlab}: %{{x:.4f}}<br>{ylab}: %{{y:.4f}}"
                "<br>|U|: %{marker.color:.4g} m/s<extra></extra>"
            ),
            name="base",
            showlegend=False,
        )
    )

    # Build line segments for arrows: base -> tip with nan separators.
    segs_x: list = []
    segs_y: list = []
    for i in idx:
        x0, y0 = float(plot_x[i]), float(plot_y[i])
        dx, dy = float(uc[i]) * scale, float(vc[i]) * scale
        segs_x.extend([x0, x0 + dx, None])
        segs_y.extend([y0, y0 + dy, None])

    fig.add_trace(
        go.Scattergl(
            x=segs_x,
            y=segs_y,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.4)", width=0.8),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title),
            xaxis=dict(title=xlab, scaleanchor="y", scaleratio=1,
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            yaxis=dict(title=ylab,
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        )
    )
    return fig


def plot_shear_rate_field(
    vtk_dir: str,
    time_step: Optional[str] = None,
    title: str = "Shear rate field",
) -> go.Figure:
    """Shear rate colormap on a 2-D slice.

    Uses the *Inferno* colormap with log-scale colouring.  If the
    ``shearRate`` field is not present in the VTK data, we attempt to
    compute it from the velocity gradient.  Falls back to a placeholder
    if neither approach works.
    """
    if not _HAS_PYVISTA:
        return _placeholder_figure(
            "pyvista is not installed -- cannot read VTK slices.", title
        )

    data = read_vtk_slice(vtk_dir, time_step)
    if not data:
        return _placeholder_figure("No VTK slice data found.", title)

    if "shearRate" in data:
        sr = data["shearRate"]
    elif "U" in data and data["U"].ndim == 2 and data["U"].shape[1] >= 3:
        # Approximate shear rate from velocity magnitude gradient using
        # local finite differences.  This is a rough surrogate -- proper
        # shear rate requires the full tensor gradient.
        u_arr = data["U"]
        mag = np.linalg.norm(u_arr[:, :3], axis=1)
        pts = data["points"]

        from scipy.spatial import cKDTree  # type: ignore[import-untyped]

        tree = cKDTree(pts)
        _, nn_idx = tree.query(pts, k=min(7, len(pts)))
        sr = np.zeros(len(pts))
        for i in range(len(pts)):
            neighbours = nn_idx[i, 1:]
            dists = np.linalg.norm(pts[neighbours] - pts[i], axis=1)
            dvals = np.abs(mag[neighbours] - mag[i])
            valid = dists > 1e-30
            if valid.any():
                sr[i] = float(np.mean(dvals[valid] / dists[valid]))
    else:
        return _placeholder_figure(
            "shearRate field not available in slice data.\n"
            "Ensure 'shearRate' is sampled by the surfaces function object,\n"
            "or velocity (U) is present for gradient estimation.",
            title,
        )

    return _scatter_colormap(
        data, sr, colorscale="Inferno", cbar_title="shearRate [1/s]",
        title=title, log_scale=True,
    )


def plot_tracer_field(
    vtk_dir: str,
    time_step: Optional[str] = None,
    title: str = "Tracer concentration",
) -> go.Figure:
    """Tracer concentration colormap on a 2-D slice.

    Uses the ``RdBu_r`` diverging colormap centred on the mean value.
    """
    if not _HAS_PYVISTA:
        return _placeholder_figure(
            "pyvista is not installed -- cannot read VTK slices.", title
        )

    data = read_vtk_slice(vtk_dir, time_step)
    if not data:
        return _placeholder_figure("No VTK slice data found.", title)

    if "tracer" not in data:
        return _placeholder_figure(
            "Tracer field not found in slice data.", title
        )

    tracer = data["tracer"]
    return _scatter_colormap(
        data, tracer, colorscale="RdBu_r", cbar_title="tracer [-]",
        title=title,
    )


# ===================================================================
# Time-series parsing
# ===================================================================

def parse_openfoam_dat(filepath: str) -> Tuple[List[float], Dict[str, List[float]]]:
    """Parse an OpenFOAM function-object ``.dat`` file.

    Lines starting with ``#`` are treated as comments.  The first
    non-comment column is assumed to be time; subsequent columns are
    named ``col1``, ``col2``, ...  If a header comment contains
    tab-separated field names these are used instead.

    Parameters
    ----------
    filepath : str
        Path to the ``.dat`` file.

    Returns
    -------
    tuple[list[float], dict[str, list[float]]]
        ``(times, values_dict)`` where *values_dict* maps column names
        to lists of float values, one per timestep.
    """
    fpath = Path(filepath)
    if not fpath.is_file():
        return [], {}

    header_names: List[str] = []
    times: List[float] = []
    columns: Dict[str, List[float]] = {}

    with open(fpath) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # Try to extract column names from the last comment line
                # that looks like a header (tab- or multi-space-separated).
                stripped = line.lstrip("# ").strip()
                parts = re.split(r"\t+|\s{2,}", stripped)
                if len(parts) >= 2:
                    header_names = parts
                continue

            # Data line -- may contain parenthesised vectors.
            # Strip parentheses and commas so all values are plain floats.
            cleaned = line.replace("(", " ").replace(")", " ").replace(",", " ")
            parts = cleaned.split()

            try:
                vals = [float(v) for v in parts]
            except ValueError:
                continue

            if len(vals) < 2:
                continue

            times.append(vals[0])

            # Assign column names.
            for ci, v in enumerate(vals[1:], start=1):
                if header_names and ci < len(header_names):
                    col_name = header_names[ci]
                else:
                    col_name = f"col{ci}"
                columns.setdefault(col_name, []).append(v)

    return times, columns


def _parse_moment_dat(filepath: str) -> Tuple[List[float], List[float]]:
    """Parse an OpenFOAM ``moment.dat`` file that uses parenthesised vectors.

    The format is typically::

        # Time (px py pz) (vx vy vz) (porousx porousy porousz)
        0.001 ((0 0 0.123) (0 0 0.00456) (0 0 0))

    Returns ``(times, total_torque_z)`` where the z-torque is the sum of
    pressure and viscous z-components (absolute value).
    """
    fpath = Path(filepath)
    if not fpath.is_file():
        return [], []

    times: List[float] = []
    torques: List[float] = []

    with open(fpath) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Try the parenthesised format first.
            m = re.match(
                r"([\d.eE+\-]+)\s+\(\(([^)]+)\)\s+\(([^)]+)\)", line,
            )
            if m:
                t = float(m.group(1))
                pressure = [float(x) for x in m.group(2).split()]
                viscous = [float(x) for x in m.group(3).split()]
                tz = abs(pressure[2] + viscous[2]) if len(pressure) > 2 and len(viscous) > 2 else 0.0
                times.append(t)
                torques.append(tz)
                continue

            # Fall back to plain-column format:
            #   time  px  py  pz  vx  vy  vz
            cleaned = line.replace("(", " ").replace(")", " ")
            parts = cleaned.split()
            try:
                vals = [float(v) for v in parts]
            except ValueError:
                continue
            if len(vals) >= 7:
                t = vals[0]
                pz = vals[3]
                vz = vals[6]
                times.append(t)
                torques.append(abs(pz + vz))

    return times, torques


def _parse_probe_file(filepath: str) -> Tuple[List[float], List[List[float]]]:
    """Parse an OpenFOAM probe output file.

    Returns ``(times, columns)`` where *columns* is a list of per-probe
    value lists.
    """
    fpath = Path(filepath)
    if not fpath.is_file():
        return [], []

    n_probes: Optional[int] = None
    times: List[float] = []
    columns: List[List[float]] = []

    with open(fpath) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Extract probe count from header if available.
                m = re.search(r"Probe\s+(\d+)", line)
                if m and n_probes is None:
                    # Count probe header lines to determine n_probes later.
                    pass
                continue

            parts = line.split()
            try:
                vals = [float(v) for v in parts]
            except ValueError:
                continue

            if len(vals) < 2:
                continue

            times.append(vals[0])
            if not columns:
                columns = [[] for _ in range(len(vals) - 1)]
            for ci, v in enumerate(vals[1:]):
                if ci < len(columns):
                    columns[ci].append(v)

    return times, columns


# ===================================================================
# Time-series plots
# ===================================================================

def plot_mixing_cov(postproc_dir: str) -> go.Figure:
    """CoV of tracer concentration versus time.

    Reads ``mixingCoV/0/volFieldValue.dat`` (or the latest time
    subdirectory).

    Parameters
    ----------
    postproc_dir : str
        Path to the ``postProcessing`` directory.
    """
    title = "Mixing homogeneity (CoV of tracer)"
    filepath = _find_dat_file(postproc_dir, "mixingCoV", "volFieldValue.dat")
    if filepath is None:
        return _placeholder_figure("mixingCoV data not found.", title)

    times, cols = parse_openfoam_dat(str(filepath))
    if not times:
        return _placeholder_figure("No data in mixingCoV file.", title)

    # First value column is the CoV.
    cov_values = list(cols.values())[0] if cols else []
    if not cov_values:
        return _placeholder_figure("No CoV values parsed.", title)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=cov_values,
            mode="lines",
            line=dict(color="rgb(0,200,220)", width=2),
            name="CoV",
            hovertemplate="t = %{x:.3f} s<br>CoV = %{y:.4f}<extra></extra>",
        )
    )

    # Reference line at CoV = 0.05 (well-mixed threshold).
    fig.add_hline(
        y=0.05, line_dash="dash", line_color="rgb(255,100,100)",
        annotation_text="CoV = 0.05", annotation_font_color="rgb(255,100,100)",
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title),
            xaxis=dict(title="Time [s]",
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            yaxis=dict(title="CoV [-]",
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        )
    )
    return fig


def plot_torque_history(postproc_dir: str) -> go.Figure:
    """Torque (z-component) versus time from ``impellerForces/0/moment.dat``.

    Parameters
    ----------
    postproc_dir : str
        Path to the ``postProcessing`` directory.
    """
    title = "Impeller torque vs time"
    filepath = _find_dat_file(postproc_dir, "impellerForces", "moment.dat")
    if filepath is None:
        return _placeholder_figure("impellerForces/moment.dat not found.", title)

    times, torques = _parse_moment_dat(str(filepath))
    if not times:
        return _placeholder_figure("No torque data parsed.", title)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=torques,
            mode="lines",
            line=dict(color="rgb(255,170,50)", width=2),
            name="Torque (z)",
            hovertemplate="t = %{x:.3f} s<br>Torque = %{y:.4g} N m<extra></extra>",
        )
    )

    # Running average over second half.
    if len(torques) > 10:
        half = len(torques) // 2
        avg = sum(torques[half:]) / len(torques[half:])
        fig.add_hline(
            y=avg, line_dash="dot", line_color="rgb(255,255,100)",
            annotation_text=f"avg = {avg:.4g} N m",
            annotation_font_color="rgb(255,255,100)",
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title),
            xaxis=dict(title="Time [s]",
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            yaxis=dict(title="Torque [N m]",
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        )
    )
    return fig


def plot_shear_rate_history(postproc_dir: str) -> go.Figure:
    """Max and average shear rate versus time (dual y-axis).

    Reads ``maxShearRate/0/volFieldValue.dat`` and
    ``volumeAverages/0/volFieldValue.dat``.

    Parameters
    ----------
    postproc_dir : str
        Path to the ``postProcessing`` directory.
    """
    title = "Shear rate vs time"
    from plotly.subplots import make_subplots  # noqa: E402

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    has_data = False

    # --- Max shear rate ---
    max_fp = _find_dat_file(postproc_dir, "maxShearRate", "volFieldValue.dat")
    if max_fp is not None:
        times_max, cols_max = parse_openfoam_dat(str(max_fp))
        max_vals = list(cols_max.values())[0] if cols_max else []
        if times_max and max_vals:
            has_data = True
            fig.add_trace(
                go.Scattergl(
                    x=times_max,
                    y=max_vals,
                    mode="lines",
                    line=dict(color="rgb(255,80,80)", width=2),
                    name="Max shear rate",
                    hovertemplate="t = %{x:.3f} s<br>max = %{y:.1f} 1/s<extra></extra>",
                ),
                secondary_y=False,
            )

    # --- Average shear rate ---
    avg_fp = _find_dat_file(postproc_dir, "volumeAverages", "volFieldValue.dat")
    if avg_fp is not None:
        times_avg, cols_avg = parse_openfoam_dat(str(avg_fp))
        # volumeAverages may contain both shearRate and edr columns.
        # Take the first column as shearRate average.
        avg_vals = list(cols_avg.values())[0] if cols_avg else []
        if times_avg and avg_vals:
            has_data = True
            fig.add_trace(
                go.Scattergl(
                    x=times_avg,
                    y=avg_vals,
                    mode="lines",
                    line=dict(color="rgb(100,200,255)", width=2),
                    name="Avg shear rate",
                    hovertemplate="t = %{x:.3f} s<br>avg = %{y:.2f} 1/s<extra></extra>",
                ),
                secondary_y=True,
            )

    if not has_data:
        return _placeholder_figure("No shear rate time-series data found.", title)

    fig.update_layout(
        **_base_layout(
            title=dict(text=title),
            xaxis=dict(title="Time [s]",
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        )
    )
    fig.update_yaxes(
        title_text="Max shear rate [1/s]", secondary_y=False,
        gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
        title_font=dict(color="rgb(255,80,80)"),
        tickfont=dict(color="rgb(255,80,80)"),
    )
    fig.update_yaxes(
        title_text="Avg shear rate [1/s]", secondary_y=True,
        gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
        title_font=dict(color="rgb(100,200,255)"),
        tickfont=dict(color="rgb(100,200,255)"),
    )
    return fig


def plot_probe_data(postproc_dir: str) -> go.Figure:
    """Tracer concentration at each probe location versus time.

    Reads ``tracerProbes/0/tracer``.

    Parameters
    ----------
    postproc_dir : str
        Path to the ``postProcessing`` directory.
    """
    title = "Tracer at probe locations"
    filepath = _find_dat_file(postproc_dir, "tracerProbes", "tracer")
    if filepath is None:
        return _placeholder_figure("tracerProbes/tracer not found.", title)

    times, columns = _parse_probe_file(str(filepath))
    if not times or not columns:
        return _placeholder_figure("No probe data parsed.", title)

    # Colour palette for probes.
    palette = [
        "rgb(0,200,220)",
        "rgb(255,170,50)",
        "rgb(120,220,120)",
        "rgb(255,100,100)",
        "rgb(180,130,255)",
        "rgb(255,210,100)",
        "rgb(100,255,200)",
        "rgb(255,150,200)",
        "rgb(150,200,255)",
        "rgb(200,200,100)",
    ]

    fig = go.Figure()
    for pi, probe_vals in enumerate(columns):
        color = palette[pi % len(palette)]
        fig.add_trace(
            go.Scattergl(
                x=times,
                y=probe_vals,
                mode="lines",
                line=dict(color=color, width=1.5),
                name=f"Probe {pi + 1}",
                hovertemplate=(
                    f"Probe {pi + 1}<br>t = %{{x:.3f}} s"
                    f"<br>tracer = %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title),
            xaxis=dict(title="Time [s]",
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            yaxis=dict(title="Tracer concentration [-]",
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            legend=dict(font=dict(color=TEXT_COLOR)),
        )
    )
    return fig


# ===================================================================
# Histograms (from volume / slice data)
# ===================================================================

def plot_strain_rate_histogram(
    vtk_dir: str,
    time_step: Optional[str] = None,
) -> go.Figure:
    """Strain (shear) rate distribution from slice data.

    Displays a histogram with annotations for mean, P95, and max values.
    If full volume data is not available the slice serves as a
    representative sample.

    Parameters
    ----------
    vtk_dir : str
        Path to a VTK slice directory.
    time_step : str, optional
        Timestep to read.  Defaults to the latest.
    """
    title = "Strain rate distribution"

    if not _HAS_PYVISTA:
        return _placeholder_figure(
            "pyvista is not installed -- cannot read VTK slices.", title
        )

    data = read_vtk_slice(vtk_dir, time_step)
    if not data:
        return _placeholder_figure("No VTK slice data found.", title)

    if "shearRate" in data:
        sr = data["shearRate"].ravel()
    elif "U" in data and data["U"].ndim == 2 and data["U"].shape[1] >= 3:
        # Rough estimate from velocity magnitude gradient.
        mag = np.linalg.norm(data["U"][:, :3], axis=1)
        pts = data["points"]
        try:
            from scipy.spatial import cKDTree  # type: ignore[import-untyped]

            tree = cKDTree(pts)
            _, nn_idx = tree.query(pts, k=min(7, len(pts)))
            sr = np.zeros(len(pts))
            for i in range(len(pts)):
                neighbours = nn_idx[i, 1:]
                dists = np.linalg.norm(pts[neighbours] - pts[i], axis=1)
                dvals = np.abs(mag[neighbours] - mag[i])
                valid = dists > 1e-30
                if valid.any():
                    sr[i] = float(np.mean(dvals[valid] / dists[valid]))
        except ImportError:
            return _placeholder_figure(
                "shearRate field not in slice and scipy unavailable\n"
                "for gradient estimation.",
                title,
            )
    else:
        return _placeholder_figure(
            "shearRate field not available in slice data.", title
        )

    sr_positive = sr[sr > 0]
    if len(sr_positive) == 0:
        return _placeholder_figure("All shear rate values are zero.", title)

    mean_val = float(np.mean(sr_positive))
    p95_val = float(np.percentile(sr_positive, 95))
    max_val = float(np.max(sr_positive))

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=np.log10(sr_positive),
            nbinsx=80,
            marker_color="rgb(180,100,255)",
            marker_line=dict(color="rgb(120,60,180)", width=0.5),
            opacity=0.85,
            name="shearRate",
            hovertemplate="log10(SR) = %{x:.2f}<br>count = %{y}<extra></extra>",
        )
    )

    # Annotation lines.
    for val, label, color in [
        (mean_val, "mean", "rgb(100,255,200)"),
        (p95_val, "P95", "rgb(255,255,100)"),
        (max_val, "max", "rgb(255,100,100)"),
    ]:
        if val > 0:
            fig.add_vline(
                x=np.log10(val), line_dash="dash", line_color=color,
                annotation_text=f"{label} = {val:.1f} 1/s",
                annotation_font_color=color,
                annotation_font_size=13,
            )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title),
            xaxis=dict(title="log10( shear rate [1/s] )",
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            yaxis=dict(title="Count",
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            bargap=0.02,
        )
    )
    return fig


# ===================================================================
# Animation
# ===================================================================

def create_tracer_animation(
    vtk_dir: str,
    max_frames: int = 50,
) -> go.Figure:
    """Animated tracer evolution on a slice using plotly animation frames.

    Parameters
    ----------
    vtk_dir : str
        Path to a VTK slice directory (e.g.
        ``<case>/postProcessing/horizontalSlice``).
    max_frames : int
        Maximum number of animation frames.  If more timesteps exist
        they are evenly subsampled.
    """
    title = "Tracer evolution"

    if not _HAS_PYVISTA:
        return _placeholder_figure(
            "pyvista is not installed -- cannot create animation.", title
        )

    all_ts = get_available_timesteps(vtk_dir)
    if not all_ts:
        return _placeholder_figure("No VTK timesteps found.", title)

    # Subsample timesteps.
    if len(all_ts) > max_frames:
        indices = np.linspace(0, len(all_ts) - 1, max_frames, dtype=int)
        selected_ts = [all_ts[i] for i in indices]
    else:
        selected_ts = all_ts

    # Pre-read first frame to set up axes and colour range.
    first_data = read_vtk_slice(vtk_dir, selected_ts[0])
    if not first_data or "tracer" not in first_data:
        return _placeholder_figure("Tracer field not found in slice data.", title)

    pts0 = first_data["points"]
    tracer0 = first_data["tracer"]

    x0, y0, z0 = pts0[:, 0], pts0[:, 1], pts0[:, 2]
    z_range = float(z0.max() - z0.min())
    x_range = float(x0.max() - x0.min())
    y_range = float(y0.max() - y0.min())

    if z_range < 0.1 * max(x_range, y_range, 1e-12):
        xi, yi = 0, 1
        xlab, ylab = "x [m]", "y [m]"
    elif y_range < 0.1 * max(x_range, z_range, 1e-12):
        xi, yi = 0, 2
        xlab, ylab = "x [m]", "z [m]"
    else:
        xi, yi = 1, 2
        xlab, ylab = "y [m]", "z [m]"

    # Determine global colour range across all selected frames (sample
    # first, middle, and last).
    sample_indices = {0, len(selected_ts) // 2, len(selected_ts) - 1}
    global_min = float(np.nanmin(tracer0))
    global_max = float(np.nanmax(tracer0))
    for si in sample_indices:
        d = read_vtk_slice(vtk_dir, selected_ts[si])
        if d and "tracer" in d:
            global_min = min(global_min, float(np.nanmin(d["tracer"])))
            global_max = max(global_max, float(np.nanmax(d["tracer"])))

    # Build initial trace.
    initial_trace = go.Scattergl(
        x=pts0[:, xi],
        y=pts0[:, yi],
        mode="markers",
        marker=dict(
            color=tracer0,
            colorscale="RdBu_r",
            cmin=global_min,
            cmax=global_max,
            size=3,
            colorbar=dict(
                title=dict(text="tracer [-]", font=dict(color=TEXT_COLOR)),
                tickfont=dict(color=TEXT_COLOR),
            ),
        ),
        hovertemplate=(
            f"{xlab}: %{{x:.4f}}<br>{ylab}: %{{y:.4f}}"
            "<br>tracer: %{marker.color:.4f}<extra></extra>"
        ),
    )

    # Build frames.
    frames: List[go.Frame] = []
    for ts_name in selected_ts:
        d = read_vtk_slice(vtk_dir, ts_name)
        if not d or "tracer" not in d:
            continue
        pts_f = d["points"]
        frames.append(
            go.Frame(
                data=[
                    go.Scattergl(
                        x=pts_f[:, xi],
                        y=pts_f[:, yi],
                        mode="markers",
                        marker=dict(
                            color=d["tracer"],
                            colorscale="RdBu_r",
                            cmin=global_min,
                            cmax=global_max,
                            size=3,
                        ),
                    )
                ],
                name=ts_name,
            )
        )

    fig = go.Figure(data=[initial_trace], frames=frames)

    # Slider and play/pause buttons.
    sliders = [
        dict(
            active=0,
            steps=[
                dict(
                    method="animate",
                    args=[
                        [f.name],
                        dict(
                            mode="immediate",
                            frame=dict(duration=100, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                    label=f.name,
                )
                for f in frames
            ],
            currentvalue=dict(
                prefix="t = ",
                suffix=" s",
                font=dict(color=TEXT_COLOR, size=14),
            ),
            pad=dict(t=40),
            font=dict(color=TEXT_COLOR),
            bgcolor=GRID_COLOR,
            activebgcolor="rgb(80,80,80)",
        )
    ]

    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=100, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0),
                        ),
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
            direction="left",
            pad=dict(r=10, t=65),
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top",
            font=dict(color=TEXT_COLOR),
            bgcolor=GRID_COLOR,
        )
    ]

    fig.update_layout(
        **_base_layout(
            title=dict(text=title),
            xaxis=dict(title=xlab, scaleanchor="y", scaleratio=1,
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            yaxis=dict(title=ylab,
                       gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            sliders=sliders,
            updatemenus=updatemenus,
        )
    )
    return fig


# ===================================================================
# Internal helpers
# ===================================================================

def _find_dat_file(
    postproc_dir: str,
    func_name: str,
    filename: str,
) -> Optional[Path]:
    """Locate a function-object output file, searching through time
    subdirectories.

    Tries ``<postproc_dir>/<func_name>/0/<filename>`` first, then
    falls back to the latest numeric subdirectory.
    """
    base = Path(postproc_dir) / func_name

    # Direct check under "0".
    candidate = base / "0" / filename
    if candidate.is_file():
        return candidate

    # Scan for numeric subdirectories and try latest.
    if not base.is_dir():
        return None

    time_dirs: List[Tuple[float, Path]] = []
    for entry in base.iterdir():
        if entry.is_dir():
            try:
                t = float(entry.name)
                time_dirs.append((t, entry))
            except ValueError:
                continue

    # Try from the latest time directory backwards.
    time_dirs.sort(key=lambda x: x[0], reverse=True)
    for _, td in time_dirs:
        candidate = td / filename
        if candidate.is_file():
            return candidate

    return None
