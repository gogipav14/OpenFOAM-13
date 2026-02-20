"""
Generate createNonConformalCouplesDict for AMI interfaces.

Creates non-conformal cyclic couples at the top and bottom
of the impeller zone for sliding mesh rotation.
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
    object      createNonConformalCouplesDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""


def generate(params: dict = None) -> str:
    """Generate createNonConformalCouplesDict.

    Parameters
    ----------
    params : dict or None
        No parameters needed; the structure is always the same.
    """
    return f"""{HEADER}

fields          (p U tracer);

nonConformalCouples
{{
    // Bottom interface: below impeller zone
    impeller_bottom
    {{
        patches         (AMI_bottom AMI_bottom);
        transform       none;
    }}

    // Top interface: above impeller zone
    impeller_top
    {{
        patches         (AMI_top AMI_top);
        transform       none;
    }}
}}

// ************************************************************************* //
"""
