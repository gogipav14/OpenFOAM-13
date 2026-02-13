"""
Generate blockMeshDict for a rectangular stirred tank mesh.

Produces a simple hex mesh with 3 vertical sections
(bottom stationary, impeller rotating, top stationary),
each containing a single hex block (3 blocks total).
"""


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
    """Generate a blockMeshDict for a rectangular tank.

    Parameters
    ----------
    params : dict
        tank_length : float - tank length (x-direction) in metres
        tank_width : float - tank width (y-direction) in metres
        liquid_level : float - liquid height in metres
        zone_bottom : float - bottom of impeller zone (m)
        zone_top : float - top of impeller zone (m)
        n_x : int - cells in x-direction
        n_y : int - cells in y-direction
        n_axial_low : int - axial cells in bottom section
        n_axial_mid : int - axial cells in impeller section
        n_axial_top : int - axial cells in top section
    """
    L = params["tank_length"]
    W = params["tank_width"]
    H = params["liquid_level"]
    z1 = params["zone_bottom"]
    z2 = params["zone_top"]

    n_x = params["n_x"]
    n_y = params["n_y"]
    n_axial_low = params["n_axial_low"]
    n_axial_mid = params["n_axial_mid"]
    n_axial_top = params["n_axial_top"]

    # Half-lengths for centering the tank at origin
    hx = L / 2.0
    hy = W / 2.0

    # Four height levels
    z0 = 0.0
    heights = [z0, z1, z2, H]

    # 4 planes x 4 corners = 16 vertices
    # Per plane (looking down, CCW from +x+y quadrant toward -x):
    #   0: (-hx, -hy, z)
    #   1: ( hx, -hy, z)
    #   2: ( hx,  hy, z)
    #   3: (-hx,  hy, z)
    def v(plane, local):
        return plane * 4 + local

    vertex_lines = []
    for plane_idx, z in enumerate(heights):
        base = plane_idx * 4
        vertex_lines.append(f"    // Plane {plane_idx}: z = {z}")
        vertex_lines.append(f"    ({-hx:.8f} {-hy:.8f} {z})    // {base}")
        vertex_lines.append(f"    ( {hx:.8f} {-hy:.8f} {z})    // {base+1}")
        vertex_lines.append(f"    ( {hx:.8f}  {hy:.8f} {z})    // {base+2}")
        vertex_lines.append(f"    ({-hx:.8f}  {hy:.8f} {z})    // {base+3}")
        vertex_lines.append("")

    vertices_str = "\n".join(vertex_lines)

    # Blocks: 3 sections, 1 hex block each
    sections = [
        (0, 1, n_axial_low, ""),
        (1, 2, n_axial_mid, "impeller"),
        (2, 3, n_axial_top, ""),
    ]
    section_names = ["Bottom", "Impeller", "Top"]

    block_lines = []
    for sec_idx, (bp, tp, nax, zone) in enumerate(sections):
        tag = f" {zone}" if zone else ""
        block_lines.append(
            f"    // {section_names[sec_idx]} section "
            f"(z={heights[bp]} to z={heights[tp]})"
        )
        block_lines.append(
            f"    hex ({v(bp,0)} {v(bp,1)} {v(bp,2)} {v(bp,3)} "
            f"{v(tp,0)} {v(tp,1)} {v(tp,2)} {v(tp,3)}){tag} "
            f"({n_x} {n_y} {nax}) simpleGrading (1 1 1)"
        )
        block_lines.append("")

    blocks_str = "\n".join(block_lines)

    # Boundary faces helper
    def face(a, b, c, d):
        return f"            ({a} {b} {c} {d})"

    # Bottom (z=0 plane, normal -z)
    bottom_faces = face(v(0,0), v(0,3), v(0,2), v(0,1))

    # Top (z=H plane, normal +z)
    top_faces = face(v(3,0), v(3,1), v(3,2), v(3,3))

    # Walls: 4 side faces per section (front/back/left/right)
    wall_faces = []
    for bp, tp in [(0, 1), (1, 2), (2, 3)]:
        # Front (-y face)
        wall_faces.append(face(v(bp,0), v(bp,1), v(tp,1), v(tp,0)))
        # Right (+x face)
        wall_faces.append(face(v(bp,1), v(bp,2), v(tp,2), v(tp,1)))
        # Back (+y face)
        wall_faces.append(face(v(bp,2), v(bp,3), v(tp,3), v(tp,2)))
        # Left (-x face)
        wall_faces.append(face(v(bp,3), v(bp,0), v(tp,0), v(tp,3)))

    wall_faces_str = "\n".join(wall_faces)

    # AMI_bottom: interface at z1 (top of bottom section = bottom of impeller)
    ami_bottom_face = face(v(1,0), v(1,3), v(1,2), v(1,1))

    # AMI_top: interface at z2 (top of impeller section = bottom of top section)
    ami_top_face = face(v(2,0), v(2,1), v(2,2), v(2,3))

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
    // No curved edges for rectangular tank
);

boundary
(
    bottom
    {{
        type wall;
        faces
        (
{bottom_faces}
        );
    }}

    top
    {{
        type patch;
        faces
        (
{top_faces}
        );
    }}

    walls
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
{ami_bottom_face}
        );
    }}

    AMI_top
    {{
        type patch;
        faces
        (
{ami_top_face}
        );
    }}
);

// ************************************************************************* //
"""
