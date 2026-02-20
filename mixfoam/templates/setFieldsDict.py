"""
Generate setFieldsDict for initialising the tracer field.

For cylindrical tanks: uses cylinderToCell to set the tracer in
the bottom fraction of the liquid level.
For rectangular tanks: uses boxToCell for the same purpose.
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
    object      setFieldsDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""


def generate(params: dict) -> str:
    """Generate setFieldsDict for tracer initialisation.

    Parameters
    ----------
    params : dict
        tank_type : str - "Cylindrical" or "Rectangular"
        tank_radius : float - radius (for cylindrical)
        tank_length : float - length in x (for rectangular)
        tank_width : float - width in y (for rectangular)
        liquid_level : float - liquid height (m)
        tracer_layer_fraction : float - bottom fraction for tracer (default 0.1)
    """
    tank_type = params["tank_type"]
    liquid_level = params["liquid_level"]
    fraction = params.get("tracer_layer_fraction", 0.1)
    tracer_top = liquid_level * fraction

    if tank_type == "Cylindrical":
        radius = params["tank_radius"]
        region_block = f"""\
    cylinderToCell
    {{
        point1  (0 0 0);
        point2  (0 0 {tracer_top:.6f});
        radius  {radius};

        fieldValues
        (
            volScalarFieldValue tracer 1
        );
    }}"""
    else:
        # Rectangular
        half_l = params["tank_length"] / 2.0
        half_w = params["tank_width"] / 2.0
        region_block = f"""\
    boxToCell
    {{
        box ({-half_l:.6f} {-half_w:.6f} 0) ({half_l:.6f} {half_w:.6f} {tracer_top:.6f});

        fieldValues
        (
            volScalarFieldValue tracer 1
        );
    }}"""

    return f"""{HEADER}

defaultFieldValues
(
    volScalarFieldValue tracer 0
);

regions
(
{region_block}
);

// ************************************************************************* //
"""
