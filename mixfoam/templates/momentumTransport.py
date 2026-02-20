"""
Generate momentumTransport dictionary for LES or RANS turbulence models.

LES: WALE model with cubeRootVol delta.
RANS: kOmegaSST model.
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
    object      momentumTransport;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""


def generate(params: dict) -> str:
    """Generate momentumTransport for LES or RANS.

    Parameters
    ----------
    params : dict
        turbulence_model : str - "LES" or "RANS"
    """
    model = params.get("turbulence_model", "LES")

    if model == "LES":
        return f"""{HEADER}

simulationType  LES;

LES
{{
    // WALE (Wall-Adapting Local Eddy-viscosity) model
    model           WALE;

    WALECoeffs
    {{
        Ck          0.094;
        Cw          0.325;
    }}

    turbulence      on;
    printCoeffs     on;

    // Filter width specification
    delta           cubeRootVol;

    cubeRootVolCoeffs
    {{
        deltaCoeff  1;
    }}
}}

// ************************************************************************* //
"""
    else:
        return f"""{HEADER}

simulationType  RAS;

RAS
{{
    model           kOmegaSST;

    turbulence      on;
    printCoeffs     on;
}}

// ************************************************************************* //
"""
