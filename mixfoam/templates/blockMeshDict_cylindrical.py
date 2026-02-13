"""
Generate blockMeshDict for a cylindrical (O-grid) stirred tank mesh.

Produces an O-grid cylindrical mesh with 3 vertical sections
(bottom stationary, impeller rotating, top stationary), each containing
5 blocks (1 central + 4 outer) for a total of 15 blocks.
"""

import math


HEADER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  13
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""


def generate(params: dict) -> str:
    """Generate a blockMeshDict for a cylindrical tank with O-grid topology.

    Parameters
    ----------
    params : dict
        tank_radius : float - tank radius in metres
        liquid_level : float - liquid height in metres
        bottom_depth : float - depth below z=0 (unused, z=0 is tank bottom)
        zone_bottom : float - bottom of impeller zone (m)
        zone_top : float - top of impeller zone (m)
        ri : float - inner O-grid radius (m)
        n_radial : int - cells in radial direction (outer blocks)
        n_axial_low : int - axial cells in bottom section
        n_axial_mid : int - axial cells in impeller section
        n_axial_top : int - axial cells in top section
        n_circ : int - cells in circumferential direction
    """
    R = params["tank_radius"]
    H = params["liquid_level"]
    Hi1 = params["zone_bottom"]
    Hi2 = params["zone_top"]
    ri = params["ri"]

    n_radial = params["n_radial"]
    n_axial_low = params["n_axial_low"]
    n_axial_mid = params["n_axial_mid"]
    n_axial_top = params["n_axial_top"]
    n_circ = params["n_circ"]

    # Inner square vertices (O-grid centre)
    cos45 = math.cos(math.radians(45))
    sin45 = math.sin(math.radians(45))
    xi = ri * cos45
    yi = ri * sin45

    # Outer arc interpolation points
    xo = R * cos45
    yo = R * sin45

    # Four height levels
    z0 = 0.0
    z1 = Hi1
    z2 = Hi2
    z3 = H

    heights = [z0, z1, z2, z3]

    # Build vertex list: 8 vertices per plane, 4 planes = 32 vertices
    # Ordering per plane:
    #   0: ( xi, -yi, z)
    #   1: ( xi,  yi, z)
    #   2: (-xi,  yi, z)
    #   3: (-xi, -yi, z)
    #   4: ( R,   0,  z)
    #   5: ( 0,   R,  z)
    #   6: (-R,   0,  z)
    #   7: ( 0,  -R,  z)
    vertex_lines = []
    for plane_idx, z in enumerate(heights):
        base = plane_idx * 8
        vertex_lines.append(f"    // Plane {plane_idx}: z = {z}")
        vertex_lines.append(f"    ( {xi:.8f} {-yi:.8f} {z})       // {base}")
        vertex_lines.append(f"    ( {xi:.8f}  {yi:.8f} {z})       // {base+1}")
        vertex_lines.append(f"    ({-xi:.8f}  {yi:.8f} {z})       // {base+2}")
        vertex_lines.append(f"    ({-xi:.8f} {-yi:.8f} {z})       // {base+3}")
        vertex_lines.append(f"    ( {R:.8f}   0        {z})       // {base+4}")
        vertex_lines.append(f"    ( 0         {R:.8f}   {z})       // {base+5}")
        vertex_lines.append(f"    ({-R:.8f}   0        {z})       // {base+6}")
        vertex_lines.append(f"    ( 0        {-R:.8f}   {z})       // {base+7}")
        vertex_lines.append("")

    vertices_str = "\n".join(vertex_lines)

    # Helper: vertex index at (plane, local)
    def v(plane, local):
        return plane * 8 + local

    # Build blocks
    # Each section has 5 blocks:
    #   - Central: hex (3 0 1 2  3' 0' 1' 2')
    #   - Outer +x: hex (0 4 5 1  0' 4' 5' 1')
    #   - Outer +y: hex (1 5 6 2  1' 5' 6' 2')
    #   - Outer -x: hex (2 6 7 3  2' 6' 7' 3')
    #   - Outer -y: hex (3 7 4 0  3' 7' 4' 0')
    sections = [
        # (bottom_plane, top_plane, n_axial, zone_tag)
        (0, 1, n_axial_low, ""),
        (1, 2, n_axial_mid, "impeller"),
        (2, 3, n_axial_top, ""),
    ]

    section_names = ["Bottom", "Impeller", "Top"]
    block_lines = []
    for sec_idx, (bp, tp, nax, zone) in enumerate(sections):
        tag = f" {zone}" if zone else ""
        block_lines.append(f"    // {section_names[sec_idx]} section (z={heights[bp]} to z={heights[tp]})")
        # Central core
        block_lines.append(
            f"    hex ({v(bp,3)} {v(bp,0)} {v(bp,1)} {v(bp,2)} "
            f"{v(tp,3)} {v(tp,0)} {v(tp,1)} {v(tp,2)}){tag} "
            f"({n_circ} {n_circ} {nax}) simpleGrading (1 1 1)"
        )
        # Outer blocks
        outer_defs = [
            (0, 4, 5, 1),  # +x
            (1, 5, 6, 2),  # +y
            (2, 6, 7, 3),  # -x
            (3, 7, 4, 0),  # -y
        ]
        for (a, b, c, d) in outer_defs:
            block_lines.append(
                f"    hex ({v(bp,a)} {v(bp,b)} {v(bp,c)} {v(bp,d)} "
                f"{v(tp,a)} {v(tp,b)} {v(tp,c)} {v(tp,d)}){tag} "
                f"({n_radial} {n_circ} {nax}) simpleGrading (1 1 1)"
            )
        block_lines.append("")

    blocks_str = "\n".join(block_lines)

    # Build arc edges (outer ring only)
    arc_lines = []
    # Arc definitions per plane: (start_local, end_local, interp_x, interp_y)
    arc_defs = [
        (4, 5, xo, yo),
        (5, 6, -xo, yo),
        (6, 7, -xo, -yo),
        (7, 4, xo, -yo),
    ]
    plane_names = ["Bottom", "Mid-low", "Mid-high", "Top"]
    for plane_idx, z in enumerate(heights):
        arc_lines.append(f"    // {plane_names[plane_idx]} plane arcs")
        for (s, e, ix, iy) in arc_defs:
            arc_lines.append(
                f"    arc {v(plane_idx,s)} {v(plane_idx,e)} "
                f"({ix: .8f} {iy: .8f} {z})"
            )
        arc_lines.append("")

    arcs_str = "\n".join(arc_lines)

    # Build boundary patches
    # Helper for face definitions
    def face(a, b, c, d):
        return f"            ({a} {b} {c} {d})"

    # Bottom faces (plane 0, looking down: normal -z)
    bottom_faces = [
        face(v(0,3), v(0,2), v(0,1), v(0,0)),
        face(v(0,0), v(0,1), v(0,5), v(0,4)),
        face(v(0,1), v(0,2), v(0,6), v(0,5)),
        face(v(0,2), v(0,3), v(0,7), v(0,6)),
        face(v(0,3), v(0,0), v(0,4), v(0,7)),
    ]

    # Top faces (plane 3, looking up: normal +z)
    top_faces = [
        face(v(3,0), v(3,1), v(3,2), v(3,3)),
        face(v(3,0), v(3,4), v(3,5), v(3,1)),
        face(v(3,1), v(3,5), v(3,6), v(3,2)),
        face(v(3,2), v(3,6), v(3,7), v(3,3)),
        face(v(3,3), v(3,7), v(3,4), v(3,0)),
    ]

    # Wall faces (outer ring, all sections)
    wall_faces = []
    for bp, tp in [(0, 1), (1, 2), (2, 3)]:
        wall_faces.append(face(v(bp,4), v(tp,4), v(tp,5), v(bp,5)))
        wall_faces.append(face(v(bp,5), v(tp,5), v(tp,6), v(bp,6)))
        wall_faces.append(face(v(bp,6), v(tp,6), v(tp,7), v(bp,7)))
        wall_faces.append(face(v(bp,7), v(tp,7), v(tp,4), v(bp,4)))

    # AMI_bottom: top faces of bottom section = bottom faces of impeller section
    # These are at plane 1, viewed from the impeller section (looking down)
    ami_bottom_faces = [
        face(v(1,3), v(1,2), v(1,1), v(1,0)),
        face(v(1,0), v(1,1), v(1,5), v(1,4)),
        face(v(1,1), v(1,2), v(1,6), v(1,5)),
        face(v(1,2), v(1,3), v(1,7), v(1,6)),
        face(v(1,3), v(1,0), v(1,4), v(1,7)),
    ]

    # AMI_top: bottom faces of top section = top faces of impeller section
    # These are at plane 2, viewed from the impeller section (looking up)
    ami_top_faces = [
        face(v(2,0), v(2,1), v(2,2), v(2,3)),
        face(v(2,0), v(2,4), v(2,5), v(2,1)),
        face(v(2,1), v(2,5), v(2,6), v(2,2)),
        face(v(2,2), v(2,6), v(2,7), v(2,3)),
        face(v(2,3), v(2,7), v(2,4), v(2,0)),
    ]

    bottom_faces_str = "\n".join(bottom_faces)
    top_faces_str = "\n".join(top_faces)
    wall_faces_str = "\n".join(wall_faces)
    ami_bottom_faces_str = "\n".join(ami_bottom_faces)
    ami_top_faces_str = "\n".join(ami_top_faces)

    return f"""{HEADER}

convertToMeters 1;

vertices
(
{vertices_str}
);

blocks
(
{blocks_str}
);

edges
(
{arcs_str}
);

boundary
(
    bottom
    {{
        type wall;
        faces
        (
{bottom_faces_str}
        );
    }}

    top
    {{
        type patch;
        faces
        (
{top_faces_str}
        );
    }}

    wall
    {{
        type wall;
        faces
        (
{wall_faces_str}
        );
    }}

    AMI_bottom
    {{
        type patch;
        faces
        (
{ami_bottom_faces_str}
        );
    }}

    AMI_top
    {{
        type patch;
        faces
        (
{ami_top_faces_str}
        );
    }}
);

// ************************************************************************* //
"""
