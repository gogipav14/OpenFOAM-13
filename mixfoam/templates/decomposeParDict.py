"""
Generate decomposeParDict for parallel domain decomposition.

Uses scotch method with preservePatches constraint to keep
AMI interface patches together.
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
    object      decomposeParDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""


def generate(params: dict) -> str:
    """Generate decomposeParDict with scotch decomposition.

    Parameters
    ----------
    params : dict
        n_procs : int - number of subdomains / processors (default 4)
    """
    n_procs = params.get("n_procs", 4)

    return f"""{HEADER}

numberOfSubdomains  {n_procs};

method          scotch;

constraints
{{
    preservePatches
    {{
        type    preservePatches;
        patches (AMI_bottom AMI_top);
    }}
}}

// ************************************************************************* //
"""
