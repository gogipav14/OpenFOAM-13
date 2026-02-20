"""
Generate initial/boundary condition files for all fields.

Returns a dictionary mapping filename -> OpenFOAM file content for
U, p, tracer, and nut.
"""


HEADER_TEMPLATE = """\
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
    class       {field_class};
    object      {field_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""


def _header(field_class: str, field_name: str) -> str:
    """Return an OpenFOAM FoamFile header."""
    return HEADER_TEMPLATE.format(field_class=field_class, field_name=field_name)


def _generate_U(params: dict) -> str:
    """Generate velocity field (U) boundary conditions.

    Walls: noSlip
    Top: slip (free surface approximation)
    AMI: cyclicAMI
    Impeller (if present): noSlip
    """
    has_impeller = params.get("has_impeller_patch", False)

    impeller_block = ""
    if has_impeller:
        impeller_block = """
    impeller
    {
        type            noSlip;
    }
"""

    return f"""{_header("volVectorField", "U")}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    bottom
    {{
        type            noSlip;
    }}

    top
    {{
        // Free surface approximation - slip condition
        type            slip;
    }}

    wall
    {{
        type            noSlip;
    }}
{impeller_block}
    "AMI_.*"
    {{
        type            cyclicAMI;
    }}
}}

// ************************************************************************* //
"""


def _generate_p(params: dict) -> str:
    """Generate pressure field (p) boundary conditions.

    Walls: zeroGradient
    Top: fixedValue 0
    AMI: cyclicAMI
    Impeller (if present): zeroGradient
    """
    has_impeller = params.get("has_impeller_patch", False)

    impeller_block = ""
    if has_impeller:
        impeller_block = """
    impeller
    {
        type            zeroGradient;
    }
"""

    return f"""{_header("volScalarField", "p")}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    bottom
    {{
        type            zeroGradient;
    }}

    top
    {{
        // Reference pressure at top surface
        type            fixedValue;
        value           uniform 0;
    }}

    wall
    {{
        type            zeroGradient;
    }}
{impeller_block}
    "AMI_.*"
    {{
        type            cyclicAMI;
    }}
}}

// ************************************************************************* //
"""


def _generate_tracer(params: dict) -> str:
    """Generate tracer field boundary conditions.

    All walls: zeroGradient
    AMI: cyclicAMI
    """
    has_impeller = params.get("has_impeller_patch", False)

    impeller_block = ""
    if has_impeller:
        impeller_block = """
    impeller
    {
        type            zeroGradient;
    }
"""

    return f"""{_header("volScalarField", "tracer")}

// Passive scalar for mixing time measurement

dimensions      [0 0 0 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    bottom
    {{
        type            zeroGradient;
    }}

    top
    {{
        type            zeroGradient;
    }}

    wall
    {{
        type            zeroGradient;
    }}
{impeller_block}
    "AMI_.*"
    {{
        type            cyclicAMI;
    }}
}}

// ************************************************************************* //
"""


def _generate_nut(params: dict) -> str:
    """Generate turbulent viscosity (nut) boundary conditions.

    LES: nutUSpaldingWallFunction on walls, calculated on top
    RANS: nutkWallFunction on walls
    AMI: cyclicAMI
    """
    model = params.get("turbulence_model", "LES")
    has_impeller = params.get("has_impeller_patch", False)

    if model == "LES":
        wall_type = "nutUSpaldingWallFunction"
    else:
        wall_type = "nutkWallFunction"

    impeller_block = ""
    if has_impeller:
        impeller_block = f"""
    impeller
    {{
        type            {wall_type};
        value           uniform 0;
    }}
"""

    return f"""{_header("volScalarField", "nut")}

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    bottom
    {{
        type            {wall_type};
        value           uniform 0;
    }}

    top
    {{
        type            calculated;
        value           uniform 0;
    }}

    wall
    {{
        type            {wall_type};
        value           uniform 0;
    }}
{impeller_block}
    "AMI_.*"
    {{
        type            cyclicAMI;
    }}
}}

// ************************************************************************* //
"""


def generate_all(params: dict) -> dict:
    """Generate all boundary condition files.

    Parameters
    ----------
    params : dict
        turbulence_model : str - "LES" or "RANS"
        has_impeller_patch : bool - whether snappyHexMesh created an
                                    impeller wall patch

    Returns
    -------
    dict
        Mapping of filename (str) -> file content (str) for:
        "U", "p", "tracer", "nut"
    """
    return {
        "U": _generate_U(params),
        "p": _generate_p(params),
        "tracer": _generate_tracer(params),
        "nut": _generate_nut(params),
    }
