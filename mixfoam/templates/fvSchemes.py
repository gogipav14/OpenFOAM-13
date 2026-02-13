"""
Generate fvSchemes for stirred tank simulations.

Supports LES (backward time, LUST convection) and RANS
(steadyState time, upwind convection) schemes. The current
implementation always uses the LES-appropriate schemes.
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
    object      fvSchemes;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""


def generate(params: dict) -> str:
    """Generate fvSchemes for LES or RANS turbulence modelling.

    Parameters
    ----------
    params : dict
        turbulence_model : str - "LES" or "RANS"
    """
    model = params.get("turbulence_model", "LES")

    if model == "LES":
        ddt_scheme = "backward"
        div_phi_U = "Gauss LUST grad(U)"
        div_phi_tracer = "Gauss linearUpwind grad(tracer)"
        div_phi_k = "Gauss limitedLinear 1"
        div_phi_nuTilda = "Gauss limitedLinear 1"
    else:
        # RANS steady-state defaults
        ddt_scheme = "steadyState"
        div_phi_U = "Gauss upwind"
        div_phi_tracer = "Gauss upwind"
        div_phi_k = "Gauss upwind"
        div_phi_nuTilda = "Gauss upwind"

    return f"""{HEADER}

ddtSchemes
{{
    default         {ddt_scheme};
}}

gradSchemes
{{
    default         Gauss linear;
    grad(p)         Gauss linear;
    grad(U)         Gauss linear;
}}

divSchemes
{{
    default         none;

    // Momentum
    div(phi,U)      {div_phi_U};

    // Tracer
    div(phi,tracer) {div_phi_tracer};

    // Turbulence quantities
    div(phi,k)      {div_phi_k};
    div(phi,nuTilda) {div_phi_nuTilda};

    // Stress tensor
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}

wallDist
{{
    method          meshWave;
}}

// ************************************************************************* //
"""
