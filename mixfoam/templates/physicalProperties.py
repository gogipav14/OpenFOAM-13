"""
Generate physicalProperties (constant viscosity model).
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
    object      physicalProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""


def generate(params: dict) -> str:
    """Generate physicalProperties with constant viscosity.

    Parameters
    ----------
    params : dict
        nu : float - kinematic viscosity in m^2/s
    """
    nu = params["nu"]

    return f"""{HEADER}

viscosityModel  constant;

nu              {nu};  // [m^2/s] kinematic viscosity

// ************************************************************************* //
"""
