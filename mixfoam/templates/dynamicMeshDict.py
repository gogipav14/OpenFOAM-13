"""
Generate dynamicMeshDict for solid-body rotation of the impeller zone.

Produces a dynamicMeshDict that applies rotatingMotion to the
"impeller" cellZone using the solidBody motion solver.
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
    object      dynamicMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""


def generate(params: dict) -> str:
    """Generate a dynamicMeshDict for impeller rotation.

    Parameters
    ----------
    params : dict
        origin_x : float - x-coordinate of impeller shaft centre
        origin_y : float - y-coordinate of impeller shaft centre
        axis_z : float - 1.0 for bottom-mounted (CCW from above),
                        -1.0 for top-mounted
        rpm : float - rotational speed in revolutions per minute
    """
    origin_x = params.get("origin_x", 0.0)
    origin_y = params.get("origin_y", 0.0)
    axis_z = params.get("axis_z", 1.0)
    rpm = params["rpm"]

    return f"""{HEADER}

// Solid body rotation for impeller zone
mover
{{
    type            motionSolver;
    libs            ("libfvMeshMovers.so" "libfvMotionSolvers.so");

    motionSolver    solidBody;

    cellZone        impeller;

    solidBodyMotionFunction rotatingMotion;

    rotatingMotionCoeffs
    {{
        origin      ({origin_x} {origin_y} 0);
        axis        (0 0 {axis_z});
        omega       {rpm} [rpm];
    }}
}}

// ************************************************************************* //
"""
