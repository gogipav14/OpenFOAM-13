"""
Generate fvSolution for stirred tank simulations.

Supports GPU-accelerated pressure solving via OGL library or
standard CPU-based PCG with DIC preconditioner.
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
    object      fvSolution;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""


def generate(params: dict) -> str:
    """Generate fvSolution with GPU or CPU pressure solver.

    Parameters
    ----------
    params : dict
        use_gpu : bool - use OGLPCG solver for pressure (GPU) or PCG+DIC (CPU)
    """
    use_gpu = params.get("use_gpu", False)

    if use_gpu:
        p_solver_block = """\
    p
    {
        solver          OGLPCG;
        tolerance       1e-6;
        relTol          0.01;

        OGLCoeffs
        {
            precisionPolicy     FP32;
            iterativeRefinement on;
            maxRefineIters      2;
            innerTolerance      1e-4;
            cacheStructure      true;
            cacheValues         false;  // NCC interface values change each step
            debug               0;
            timing              true;
        }
    }"""
    else:
        p_solver_block = """\
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-6;
        relTol          0.01;
    }"""

    return f"""{HEADER}

solvers
{{
{p_solver_block}

    pFinal
    {{
        $p;
        relTol          0;
    }}

    "pcorr.*"
    {{
        $p;
        tolerance       0.02;
        relTol          0;
    }}

    "(U|UFinal)"
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.1;
    }}

    "(tracer|tracerFinal)"
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.01;
    }}
}}

PIMPLE
{{
    nOuterCorrectors    2;
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;

    consistent          yes;

    residualControl
    {{
        p
        {{
            tolerance   1e-4;
            relTol      0;
        }}
    }}
}}

relaxationFactors
{{
    equations
    {{
        ".*"            1;
    }}
}}

// ************************************************************************* //
"""
