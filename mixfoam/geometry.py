"""
Tank volume and liquid level calculations for all supported bottom styles.

Supports:
  - Cylindrical tanks: Flat, 2:1 Elliptical, 6%/10% Torispherical, Conical bottoms
  - Rectangular tanks: Flat bottom only (all rectangular tanks in MixIT data)
"""

import math
from typing import Tuple


def _cylinder_area(radius: float) -> float:
    return math.pi * radius ** 2


def _volume_flat_cylinder(h: float, radius: float) -> float:
    """Volume of a flat-bottom cylinder filled to height h."""
    return _cylinder_area(radius) * h


def _volume_elliptical_bottom(h: float, radius: float, depth: float) -> float:
    """
    Volume of a 2:1 elliptical bottom filled to height h.
    The elliptical bottom is a half-ellipsoid with semi-axes (radius, radius, depth).
    Volume of full half-ellipsoid = (2/3) * pi * R^2 * d
    Partial fill (h <= d): V = pi * R^2 * h^2 * (3*d - h) / (3 * d^2)
    This is the ellipsoidal cap formula.
    """
    if h <= 0:
        return 0.0
    if h >= depth:
        # Full bottom + cylinder above
        full_bottom = (2.0 / 3.0) * math.pi * radius ** 2 * depth
        return full_bottom + _cylinder_area(radius) * (h - depth)
    # Partial fill of ellipsoid bottom
    # For an oblate half-ellipsoid with z in [0, d], the cross-section at height z
    # has radius r(z) = R * sqrt(1 - ((d-z)/d)^2) = R * sqrt(z*(2*d - z)) / d
    # V(h) = integral_0^h pi * r(z)^2 dz = pi*R^2/d^2 * integral_0^h (2*d*z - z^2) dz
    #       = pi*R^2/d^2 * (d*h^2 - h^3/3)
    return math.pi * radius ** 2 / depth ** 2 * (depth * h ** 2 - h ** 3 / 3.0)


def _volume_conical_bottom(h: float, radius: float, depth: float) -> float:
    """
    Volume of a conical bottom filled to height h.
    Cone with apex at z=0, base radius R at z=depth.
    At height z (z <= depth): r(z) = R * z / depth
    V(h) = integral_0^h pi * (R*z/d)^2 dz = pi*R^2*h^3 / (3*d^2)  for h <= d
    """
    if h <= 0:
        return 0.0
    if h >= depth:
        full_cone = math.pi * radius ** 2 * depth / 3.0
        return full_cone + _cylinder_area(radius) * (h - depth)
    return math.pi * radius ** 2 * h ** 3 / (3.0 * depth ** 2)


def _volume_torispherical_bottom(h: float, radius: float, depth: float) -> float:
    """
    Volume of a torispherical bottom filled to height h.
    Torispherical heads are complex (crown + knuckle). For simplicity, we
    approximate as a spherical cap that matches the given depth.

    Spherical cap: R_sphere such that cap of height d has base radius R_tank.
    R_sphere = (R^2 + d^2) / (2*d)
    V_cap(h) = pi*h^2*(3*R_sphere - h)/3  for h <= d
    """
    if h <= 0:
        return 0.0
    if depth <= 0:
        return _volume_flat_cylinder(h, radius)
    if h >= depth:
        # Full bottom + cylinder above
        r_sphere = (radius ** 2 + depth ** 2) / (2.0 * depth)
        full_cap = math.pi * depth ** 2 * (3.0 * r_sphere - depth) / 3.0
        return full_cap + _cylinder_area(radius) * (h - depth)
    # Partial fill — scale the cap approximation
    # At height z in the cap, the cross-section radius is:
    # r(z) = sqrt(2*R_sphere*z - z^2)  (standard spherical cap)
    r_sphere = (radius ** 2 + depth ** 2) / (2.0 * depth)
    return math.pi * h ** 2 * (3.0 * r_sphere - h) / 3.0


def volume_at_level(h: float, tank_type: str, diameter: float,
                    length: float, width: float,
                    bottom_style: str, bottom_depth: float) -> float:
    """
    Compute liquid volume (m³) at a given fill height h (m).

    Parameters:
        h: fill height from bottom of vessel (m)
        tank_type: "Cylindrical" or "Rectangular"
        diameter: tank diameter (m) for cylindrical, or side length for rectangular
        length: tank length (m) for rectangular
        width: tank width (m) for rectangular
        bottom_style: "Flat", "2:1Elliptical", "6%Torispherical", "10%Torispherical", "Conical"
        bottom_depth: depth of the bottom head (m)
    """
    if h <= 0:
        return 0.0

    if tank_type == "Rectangular":
        # All rectangular tanks have flat bottoms in the MixIT data
        return length * width * h

    # Cylindrical
    radius = diameter / 2.0

    if bottom_style == "Flat" or bottom_depth <= 0:
        return _volume_flat_cylinder(h, radius)
    elif "Elliptical" in bottom_style:
        return _volume_elliptical_bottom(h, radius, bottom_depth)
    elif "Conical" in bottom_style:
        return _volume_conical_bottom(h, radius, bottom_depth)
    elif "Torispherical" in bottom_style:
        return _volume_torispherical_bottom(h, radius, bottom_depth)
    else:
        # Unknown — treat as flat
        return _volume_flat_cylinder(h, radius)


def compute_liquid_level(target_volume_m3: float, tank_type: str,
                         diameter: float, length: float, width: float,
                         straight_side: float, bottom_style: str,
                         bottom_depth: float,
                         tol: float = 1e-9, max_iter: int = 100) -> float:
    """
    Compute the liquid level (m) for a target fill volume using bisection.

    Returns the height h (m) from the bottom of the vessel.
    """
    if target_volume_m3 <= 0:
        return 0.0

    # Maximum possible height = bottom_depth + straight_side + head_depth
    # Use straight_side + bottom_depth as upper bound (ignoring head)
    h_max = bottom_depth + straight_side
    # Sanity: ensure we can hold the target volume
    v_max = volume_at_level(h_max, tank_type, diameter, length, width,
                            bottom_style, bottom_depth)
    if target_volume_m3 > v_max:
        # Extend search range
        h_max *= 2.0

    h_lo, h_hi = 0.0, h_max

    for _ in range(max_iter):
        h_mid = (h_lo + h_hi) / 2.0
        v_mid = volume_at_level(h_mid, tank_type, diameter, length, width,
                                bottom_style, bottom_depth)
        if abs(v_mid - target_volume_m3) < tol:
            return h_mid
        if v_mid < target_volume_m3:
            h_lo = h_mid
        else:
            h_hi = h_mid

    return (h_lo + h_hi) / 2.0


def compute_mesh_parameters(tank_type: str, diameter: float,
                            length: float, width: float,
                            liquid_level: float, bottom_depth: float,
                            impeller_clearance: float,
                            impeller_diameter: float,
                            target_cell_size: float = 0.003) -> dict:
    """
    Compute mesh sizing parameters for the given tank geometry.

    Returns dict with:
        n_radial, n_axial_low, n_axial_mid, n_axial_top, n_circ,
        ri (inner O-grid radius), zone_bottom, zone_top
    """
    radius = diameter / 2.0
    imp_radius = impeller_diameter / 2.0

    # Impeller zone vertical extent
    zone_half_height = max(imp_radius, 0.02)  # at least 2cm
    zone_bottom = max(impeller_clearance - zone_half_height, bottom_depth)
    zone_top = impeller_clearance + zone_half_height

    # Ensure zone doesn't exceed liquid level
    zone_top = min(zone_top, liquid_level - 0.01)

    # Inner O-grid radius for cylindrical tanks
    ri = max(imp_radius * 1.2, radius * 0.3)

    # Cell counts based on target cell size
    if tank_type == "Cylindrical":
        n_radial = max(int((radius - ri) / target_cell_size), 8)
        n_circ = max(int(2 * math.pi * radius / (4 * target_cell_size)), 16)
        # Make n_circ divisible by 4 for O-grid symmetry
        n_circ = ((n_circ + 3) // 4) * 4
    else:
        n_radial = max(int(diameter / target_cell_size), 20)
        n_circ = max(int(width / target_cell_size), 20)

    n_axial_low = max(int((zone_bottom) / target_cell_size), 4)
    n_axial_mid = max(int((zone_top - zone_bottom) / target_cell_size), 4)
    n_axial_top = max(int((liquid_level - zone_top) / target_cell_size), 4)

    return {
        "n_radial": n_radial,
        "n_axial_low": n_axial_low,
        "n_axial_mid": n_axial_mid,
        "n_axial_top": n_axial_top,
        "n_circ": n_circ,
        "ri": ri,
        "zone_bottom": zone_bottom,
        "zone_top": zone_top,
    }


def estimate_cell_count(mesh_params: dict, tank_type: str) -> int:
    """Rough estimate of total mesh cell count."""
    nr = mesh_params["n_radial"]
    nc = mesh_params["n_circ"]
    n_lo = mesh_params["n_axial_low"]
    n_mid = mesh_params["n_axial_mid"]
    n_top = mesh_params["n_axial_top"]
    n_total_axial = n_lo + n_mid + n_top

    if tank_type == "Cylindrical":
        # O-grid: center block (nc*nc) + 4 outer blocks (nr*nc)
        cells_per_layer = nc * nc + 4 * nr * nc
        return cells_per_layer * n_total_axial
    else:
        return nr * nc * n_total_axial
