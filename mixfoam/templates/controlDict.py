"""
Generate controlDict for stirred tank LES/RANS simulations.

Includes function objects for field averaging, shear rate, energy dissipation
rate (EDR), impeller forces/torque, mixing quality (CoV), tracer probes,
volume averages, and max shear rate.
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
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""


def generate(params: dict) -> str:
    """Generate a controlDict with comprehensive function objects.

    Parameters
    ----------
    params : dict
        end_time : float - simulation end time (s)
        delta_t : float - initial time step (s)
        write_interval : float - write interval (s)
        max_co : float - maximum Courant number (default 0.5)
        nu : float - kinematic viscosity (m^2/s) for EDR calculation
        density : float - fluid density (kg/m^3) for forces
        impeller_offset_x : float - CofR x for forces
        impeller_offset_y : float - CofR y for forces
        impeller_clearance : float - CofR z for forces (impeller height)
        use_gpu : bool - whether to load libOGL.so
        probe_locations : list of tuples - (x, y, z) probe positions
        has_impeller_patch : bool - whether snappyHexMesh created impeller patch
    """
    end_time = params["end_time"]
    delta_t = params["delta_t"]
    write_interval = params["write_interval"]
    max_co = params.get("max_co", 0.5)
    nu = params["nu"]
    density = params["density"]
    cofr_x = params.get("impeller_offset_x", 0.0)
    cofr_y = params.get("impeller_offset_y", 0.0)
    cofr_z = params.get("impeller_clearance", 0.1)
    use_gpu = params.get("use_gpu", False)
    probe_locations = params.get("probe_locations", [(0, 0, 0.15)])
    has_impeller_patch = params.get("has_impeller_patch", True)

    # Libs line
    if use_gpu:
        libs_str = 'libs            ("libOGL.so");'
    else:
        libs_str = "libs            ();"

    # Probe locations string
    probe_lines = []
    for loc in probe_locations:
        probe_lines.append(f"            ({loc[0]} {loc[1]} {loc[2]})")
    probes_str = "\n".join(probe_lines)

    # Impeller forces function object (only if impeller patch exists)
    if has_impeller_patch:
        impeller_forces_block = f"""
    impellerForces
    {{
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   100;
        log             true;

        patches         (impeller);
        rho             rhoInf;
        rhoInf          {density};
        CofR            ({cofr_x} {cofr_y} {cofr_z});
    }}"""
    else:
        impeller_forces_block = """
    // impellerForces: skipped (no impeller patch from snappyHexMesh)"""

    return f"""{HEADER}

{libs_str}

application     foamRun;

solver          incompressibleFluid;

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         {end_time};

deltaT          {delta_t};

writeControl    adjustableRunTime;

writeInterval   {write_interval};

purgeWrite      5;

writeFormat     binary;

writePrecision  8;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable yes;

adjustTimeStep  yes;

maxCo           {max_co};

maxDeltaT       {delta_t * 10};

functions
{{
    // ------------------------------------------------------------------
    // 1. Field averaging (U, p, tracer)
    // ------------------------------------------------------------------
    fieldAverage
    {{
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;

        fields
        (
            U
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
            p
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
            tracer
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
        );
    }}

    // ------------------------------------------------------------------
    // 2. Shear rate: mag(symm(grad(U)))
    // ------------------------------------------------------------------
    shearRate
    {{
        type            coded;
        libs            ("libutilityFunctionObjects.so");
        writeControl    writeTime;
        name            shearRate;

        codeExecute
        #{{
            const volVectorField& U =
                mesh().lookupObject<volVectorField>("U");

            volScalarField& shearRate =
                mesh().lookupObjectRef<volScalarField>("shearRate");

            shearRate = mag(symm(fvc::grad(U)));

            shearRate.correctBoundaryConditions();
        #}};

        codeWrite
        #{{
            // Written by writeControl
        #}};

        codeEnd
        #{{
        #}};
    }}

    // ------------------------------------------------------------------
    // 3. Energy dissipation rate: EDR = 2 * nu * S:S
    // ------------------------------------------------------------------
    edr
    {{
        type            coded;
        libs            ("libutilityFunctionObjects.so");
        writeControl    writeTime;
        name            edr;

        codeExecute
        #{{
            const volVectorField& U =
                mesh().lookupObject<volVectorField>("U");

            volScalarField& edr =
                mesh().lookupObjectRef<volScalarField>("edr");

            const volSymmTensorField S(symm(fvc::grad(U)));
            edr = 2.0 * {nu} * (S && S);

            edr.correctBoundaryConditions();
        #}};

        codeWrite
        #{{
        #}};

        codeEnd
        #{{
        #}};
    }}

    // ------------------------------------------------------------------
    // 4. Impeller forces (torque / power)
    // ------------------------------------------------------------------
{impeller_forces_block}

    // ------------------------------------------------------------------
    // 5. Mixing quality: CoV of tracer
    // ------------------------------------------------------------------
    mixingCoV
    {{
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   100;
        writeFields     false;
        log             true;
        operation       CoV;

        fields
        (
            tracer
        );

        regionType      all;
    }}

    // ------------------------------------------------------------------
    // 6. Tracer probes (TMB locations)
    // ------------------------------------------------------------------
    tracerProbes
    {{
        type            probes;
        libs            ("libsampling.so");
        writeControl    timeStep;
        writeInterval   10;

        probeLocations
        (
{probes_str}
        );

        fields
        (
            U
            p
            tracer
        );
    }}

    // ------------------------------------------------------------------
    // 7. Volume-averaged shear rate and EDR
    // ------------------------------------------------------------------
    volumeAverages
    {{
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   100;
        writeFields     false;
        log             true;
        operation       volAverage;

        fields
        (
            shearRate
            edr
        );

        regionType      all;
    }}

    // ------------------------------------------------------------------
    // 8. Maximum shear rate
    // ------------------------------------------------------------------
    maxShearRate
    {{
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   100;
        writeFields     false;
        log             true;
        operation       max;

        fields
        (
            shearRate
        );

        regionType      all;
    }}

    // ------------------------------------------------------------------
    // 9. Horizontal slice through impeller plane (for visualization)
    // ------------------------------------------------------------------
    horizontalSlice
    {{
        type            surfaces;
        libs            ("libsampling.so");
        writeControl    writeTime;
        surfaceFormat   vtk;
        interpolationScheme cellPoint;

        fields
        (
            U
            p
            tracer
        );

        surfaces
        (
            zSlice
            {{
                type        cuttingPlane;
                point       ({cofr_x} {cofr_y} {cofr_z});
                normal      (0 0 1);
            }}
        );
    }}

    // ------------------------------------------------------------------
    // 10. Vertical slice through impeller axis (for visualization)
    // ------------------------------------------------------------------
    verticalSlice
    {{
        type            surfaces;
        libs            ("libsampling.so");
        writeControl    writeTime;
        surfaceFormat   vtk;
        interpolationScheme cellPoint;

        fields
        (
            U
            p
            tracer
        );

        surfaces
        (
            ySlice
            {{
                type        cuttingPlane;
                point       ({cofr_x} {cofr_y} 0);
                normal      (0 1 0);
            }}
        );
    }}
}}

// ************************************************************************* //
"""
