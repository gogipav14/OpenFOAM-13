"""
Generate snappyHexMeshDict for impeller and baffle surface refinement.

Produces a snappyHexMeshDict that refines the background hex mesh around
STL geometry files for impellers and baffles located in constant/geometry/.
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
    object      snappyHexMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""


def generate(params: dict) -> str:
    """Generate a snappyHexMeshDict for impeller/baffle refinement.

    Parameters
    ----------
    params : dict
        has_impeller : bool - whether to include impeller STL
        has_baffle : bool - whether to include baffle STL(s)
        num_baffles : int - number of baffle STL files
        refine_level_impeller : int - refinement level for impeller (default 3)
        refine_level_baffle : int - refinement level for baffles (default 2)
        location_in_mesh_z : float - z-coordinate for locationInMesh point
    """
    has_impeller = params.get("has_impeller", True)
    has_baffle = params.get("has_baffle", False)
    num_baffles = params.get("num_baffles", 0)
    refine_impeller = params.get("refine_level_impeller", 3)
    refine_baffle = params.get("refine_level_baffle", 2)
    loc_z = params.get("location_in_mesh_z", 0.05)

    # Build geometry entries
    geometry_entries = []
    if has_impeller:
        geometry_entries.append(
            f"    impeller.stl\n"
            f"    {{\n"
            f"        type triSurfaceMesh;\n"
            f"        name impeller;\n"
            f"    }}"
        )
    if has_baffle:
        for i in range(num_baffles):
            geometry_entries.append(
                f"    baffle_{i}.stl\n"
                f"    {{\n"
                f"        type triSurfaceMesh;\n"
                f"        name baffle_{i};\n"
                f"    }}"
            )

    geometry_str = "\n\n".join(geometry_entries)

    # Build refinement surface entries
    refinement_entries = []
    if has_impeller:
        refinement_entries.append(
            f"        impeller\n"
            f"        {{\n"
            f"            level ({refine_impeller} {refine_impeller});\n"
            f"            patchInfo\n"
            f"            {{\n"
            f"                type wall;\n"
            f"            }}\n"
            f"        }}"
        )
    if has_baffle:
        for i in range(num_baffles):
            refinement_entries.append(
                f"        baffle_{i}\n"
                f"        {{\n"
                f"            level ({refine_baffle} {refine_baffle});\n"
                f"            patchInfo\n"
                f"            {{\n"
                f"                type wall;\n"
                f"            }}\n"
                f"        }}"
            )

    refinement_str = "\n\n".join(refinement_entries)

    # Build refinement region entries (volumetric refinement near surfaces)
    region_entries = []
    if has_impeller:
        region_entries.append(
            f"        impeller\n"
            f"        {{\n"
            f"            mode distance;\n"
            f"            levels ((0.01 {refine_impeller}));\n"
            f"        }}"
        )
    if has_baffle:
        for i in range(num_baffles):
            region_entries.append(
                f"        baffle_{i}\n"
                f"        {{\n"
                f"            mode distance;\n"
                f"            levels ((0.005 {refine_baffle}));\n"
                f"        }}"
            )

    region_str = "\n\n".join(region_entries) if region_entries else ""

    return f"""{HEADER}

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
{geometry_str}
}}

castellatedMeshControls
{{
    maxLocalCells       1000000;
    maxGlobalCells      10000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;

    features
    (
        // No explicit feature edge refinement
    );

    refinementSurfaces
    {{
{refinement_str}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
{region_str}
    }}

    locationInMesh (0 0 {loc_z});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      100;
    nRelaxIter      5;

    nFeatureSnapIter    10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes       true;
    layers
    {{
    }}
    expansionRatio      1.0;
    finalLayerThickness 0.3;
    minThickness        0.1;
    nGrow               0;
    featureAngle        60;
    nRelaxIter          3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.05;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;

    nSmoothScale        4;
    errorReduction      0.75;

    relaxed
    {{
        maxNonOrtho     75;
    }}
}}

mergeTolerance  1e-6;

// ************************************************************************* //
"""
