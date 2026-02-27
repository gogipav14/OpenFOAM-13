#!/bin/bash
# =============================================================================
# Sartorius 50L Scaling Benchmark: GPU AMG vs CPU GAMG
# =============================================================================
# Runs the Sartorius mixing case at multiple mesh sizes to generate
# the scaling data needed for Paper 1 (GPU AMG + ILU-ISAI).
#
# Mesh sizes: 73K (base), 200K, 500K, 1M, 2M, 5M
# Solver configs: CPU GAMG, GPU AMG-PCG + BJ, GPU AMG-PCG + ILU-ISAI
#
# Usage (inside Docker container):
#   source /opt/OpenFOAM-13/etc/bashrc
#   bash /workspace/benchmarks/sartorius_scaling_benchmark.sh [--quick]
#
# --quick: run only 73K and 500K for a fast smoke test
# =============================================================================

set -eo pipefail

QUICK_MODE=0
if [ "${1:-}" = "--quick" ]; then
    QUICK_MODE=1
    echo "[Quick mode: 73K and 500K only]"
fi

RESULTS_DIR="/workspace/benchmarks/sartorius_scaling_results"
BASE_CASE="/workspace/benchmarks/sartorius_50L_benchmark"
NSTEPS=10       # 10 timesteps = 40 pressure solves
DT=0.005

# Vessel dimensions (from blockMeshDict)
LX=0.41
LY=0.41
LZ=0.465

# Mesh scaling levels: name nx ny nz
# Base: 40x40x46 = 73,600 cells
# Scale factor applied uniformly to maintain aspect ratio
if [ "$QUICK_MODE" = "1" ]; then
    MESH_LEVELS=(
        "73k    40  40  46"
        "500k   80  80  80"
    )
else
    MESH_LEVELS=(
        "73k    40  40  46"
        "200k   60  60  56"
        "500k   80  80  80"
        "1M     100 100 100"
        "2M     126 126 126"
        "5M     172 172 172"
    )
fi

# Solver configurations: "name type description"
SOLVER_CONFIGS=(
    "cpuGAMG        cpu     GAMG+GaussSeidel_pressure,smoothSolver+symGS_momentum"
    "gpuAMG_BJ      gpu_bj  OGLPCG+AMG_pressure,OGLBiCGStab+BlockJacobi_momentum"
    "gpuAMG_ILU     gpu_ilu OGLPCG+AMG_pressure,OGLBiCGStab+ILU-ISAI_momentum"
)

rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "Sartorius 50L Scaling Benchmark"
echo "============================================"
echo "Vessel: ${LX}m x ${LY}m x ${LZ}m"
echo "Steps: $NSTEPS (dt=$DT)"
echo ""

# Master CSV header
echo "mesh_label,ncells,nx,ny,nz,solver,avg_p_iters,avg_pFinal_iters,avg_Ux_iters,exec_time_s,p_solve_ms,notes" \
    > "$RESULTS_DIR/scaling_results.csv"

write_fvsolution_cpu() {
    local casedir=$1

    cat > "$casedir/system/fvSolution" << 'EOFSOL'
FoamFile { format ascii; class dictionary; object fvSolution; }
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-06;
        relTol          0.01;
    }
    pFinal
    {
        $p;
        relTol          0;
    }
    "(U|k|epsilon)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }
    "(U|k|epsilon)Final"
    {
        $U;
        relTol          0;
    }
}
PIMPLE
{
    momentumPredictor yes;
    nOuterCorrectors  2;
    nCorrectors       2;
    nNonOrthogonalCorrectors 0;
    pRefCell          0;
    pRefValue         0;
}
relaxationFactors { equations { ".*" 1; } }
EOFSOL
}

write_fvsolution_gpu() {
    local casedir=$1
    local preconditioner=$2  # "blockJacobi" or "ILU"

    cat > "$casedir/system/fvSolution" << EOFSOL
FoamFile { format ascii; class dictionary; object fvSolution; }
solvers
{
    p
    {
        solver          OGLPCG;
        tolerance       1e-06;
        relTol          0.01;
        OGLCoeffs
        {
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      multigrid;
            blockSize           4;
            mgCacheInterval     10;
        }
    }
    pFinal
    {
        solver          OGLPCG;
        tolerance       1e-06;
        relTol          0;
        OGLCoeffs
        {
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               1;
            preconditioner      multigrid;
            blockSize           4;
            mgCacheInterval     10;
        }
    }
    "(U|k|epsilon)"
    {
        solver          OGLBiCGStab;
        tolerance       1e-06;
        relTol          0.1;
        OGLCoeffs
        {
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      $preconditioner;
        }
    }
    "(U|k|epsilon)Final"
    {
        solver          OGLBiCGStab;
        tolerance       1e-06;
        relTol          0;
        OGLCoeffs
        {
            precisionPolicy     FP64;
            iterativeRefinement off;
            cacheStructure      true;
            cacheValues         false;
            debug               0;
            preconditioner      $preconditioner;
        }
    }
}
PIMPLE
{
    momentumPredictor yes;
    nOuterCorrectors  2;
    nCorrectors       2;
    nNonOrthogonalCorrectors 0;
    pRefCell          0;
    pRefValue         0;
}
relaxationFactors { equations { ".*" 1; } }
EOFSOL
}

write_controldict() {
    local casedir=$1
    local solver_type=$2  # "cpu" or "gpu_bj" or "gpu_ilu"

    local endTime
    endTime=$(python3 -c "print(f'{$NSTEPS * $DT:.6f}')")

    local libs_line=""
    if [ "$solver_type" != "cpu" ]; then
        libs_line='libs ("libOGL.so");'
    fi

    cat > "$casedir/system/controlDict" << EOFCD
FoamFile { format ascii; class dictionary; object controlDict; }
$libs_line
solver          incompressibleFluid;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         $endTime;
deltaT          $DT;
writeControl    timeStep;
writeInterval   1000;
purgeWrite      1;
writeFormat     binary;
writePrecision  8;
writeCompression off;
EOFCD
}

modify_blockmeshdict() {
    local casedir=$1
    local nx=$2
    local ny=$3
    local nz=$4

    # Replace the mesh resolution variables in blockMeshDict
    sed -i "s/^nx  [0-9]*/nx  $nx/" "$casedir/system/blockMeshDict"
    sed -i "s/^ny  [0-9]*/ny  $ny/" "$casedir/system/blockMeshDict"
    sed -i "s/^nz  [0-9]*/nz  $nz/" "$casedir/system/blockMeshDict"
}

extract_metrics() {
    local output=$1
    local solver_type=$2

    # Extract pressure iteration counts
    if [ "$solver_type" = "cpu" ]; then
        avg_p=$(echo "$output" | grep "Solving for p," | tail -n +2 | \
            awk 'NR%2==1' | \
            awk -F'No Iterations ' '{s+=$2; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
        avg_pf=$(echo "$output" | grep "Solving for p," | tail -n +2 | \
            awk 'NR%2==0' | \
            awk -F'No Iterations ' '{s+=$2; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
        avg_ux=$( (echo "$output" | grep "Solving for Ux," || true) | tail -n +2 | \
            awk -F'No Iterations ' '{s+=$2; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
        solve_ms="N/A"
    else
        avg_p=$( (echo "$output" | grep "OGLPCG.*Solving for p" || true) | \
            tail -n +2 | awk 'NR%2==1' | \
            awk -F'No Iterations ' '{s+=$2; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
        avg_pf=$( (echo "$output" | grep "OGLPCG.*Solving for p" || true) | \
            tail -n +2 | awk 'NR%2==0' | \
            awk -F'No Iterations ' '{s+=$2; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
        avg_ux=$( (echo "$output" | grep "OGLBiCGStab.*Solving for Ux" || true) | tail -n +2 | \
            awk -F'No Iterations ' '{s+=$2; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
        solve_ms=$( (echo "$output" | grep "CG solve:" || true) | \
            awk -F'CG solve: ' '{print $2}' | awk '{print $1}' | tail -n +2 | \
            awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
    fi

    # Total execution time
    exec_time=$( (echo "$output" | grep "ExecutionTime" || true) | tail -1 | \
        awk -F'= ' '{print $2}' | awk '{print $1}')

    echo "${avg_p}|${avg_pf}|${avg_ux}|${exec_time}|${solve_ms}"
}

# Main benchmark loop
for mesh_spec in "${MESH_LEVELS[@]}"; do
    read -r mesh_label nx ny nz <<< "$mesh_spec"
    ncells=$((nx * ny * nz))

    echo "============================================"
    echo "Mesh: $mesh_label (${nx}x${ny}x${nz} = ${ncells} cells)"
    echo "============================================"

    # Generate mesh once per level, then copy for each solver
    meshdir="$RESULTS_DIR/_mesh_${mesh_label}"
    rm -rf "$meshdir"
    cp -r "$BASE_CASE" "$meshdir"
    find "$meshdir" -mindepth 1 -maxdepth 1 -name '[1-9]*' -type d -exec rm -rf {} + 2>/dev/null || true
    find "$meshdir" -mindepth 1 -maxdepth 1 -name '0.*' -type d -exec rm -rf {} + 2>/dev/null || true
    rm -rf "$meshdir/postProcessing"
    modify_blockmeshdict "$meshdir" "$nx" "$ny" "$nz"
    echo "  Generating mesh (${nx}x${ny}x${nz})..."
    blockMesh -case "$meshdir" > /dev/null 2>&1
    topoSet -case "$meshdir" > /dev/null 2>&1
    # Remove tracer field: the 0/ fields from the base case (73K) won't match
    # the new mesh size. Tracer is not needed for pressure/momentum benchmark.
    rm -f "$meshdir/0/tracer"
    echo "  Mesh ready: $(ls "$meshdir"/constant/polyMesh/points 2>/dev/null && echo 'OK' || echo 'FAILED')"

    for solver_spec in "${SOLVER_CONFIGS[@]}"; do
        read -r solver_name solver_type description <<< "$solver_spec"

        casedir="$RESULTS_DIR/${mesh_label}_${solver_name}"
        rm -rf "$casedir"

        echo "  --- $solver_name ($description) ---"

        # Copy pre-meshed case
        cp -r "$meshdir" "$casedir"

        # Write solver config
        if [ "$solver_type" = "cpu" ]; then
            write_fvsolution_cpu "$casedir"
        elif [ "$solver_type" = "gpu_bj" ]; then
            write_fvsolution_gpu "$casedir" "blockJacobi"
        elif [ "$solver_type" = "gpu_ilu" ]; then
            write_fvsolution_gpu "$casedir" "ILU"
        fi
        write_controldict "$casedir" "$solver_type"

        # Run solver
        echo "      Running ($NSTEPS steps)..."
        output=$(foamRun -case "$casedir" 2>&1 || true)

        # Save full log
        echo "$output" > "$RESULTS_DIR/${mesh_label}_${solver_name}.log"

        # Check for fatal errors
        if echo "$output" | grep -qi "FOAM FATAL" 2>/dev/null; then
            echo "      *** FATAL ERROR ***"
            (echo "$output" | grep -A3 "FOAM FATAL" || true) | head -10
            echo "$mesh_label,$ncells,$nx,$ny,$nz,$solver_name,FATAL,FATAL,FATAL,FATAL,FATAL,error" \
                >> "$RESULTS_DIR/scaling_results.csv"
            continue
        fi

        # Extract metrics
        metrics=$(extract_metrics "$output" "$solver_type")
        IFS='|' read -r avg_p avg_pf avg_ux exec_time solve_ms <<< "$metrics"

        echo "      p=$avg_p  pFinal=$avg_pf  Ux=$avg_ux  time=${exec_time}s"

        echo "$mesh_label,$ncells,$nx,$ny,$nz,$solver_name,$avg_p,$avg_pf,$avg_ux,$exec_time,$solve_ms," \
            >> "$RESULTS_DIR/scaling_results.csv"
    done

    # Clean up temporary mesh directory to save disk space
    rm -rf "$meshdir"
    echo ""
done

echo ""
echo "============================================"
echo "Scaling Summary"
echo "============================================"
echo ""
printf "%-8s %8s %-12s %8s %8s %8s %10s\n" \
    "Mesh" "Cells" "Solver" "p_iters" "pF_iters" "Ux_iters" "Time(s)"
printf "%s\n" "-----------------------------------------------------------------------"

while IFS=',' read -r label ncells nx ny nz solver avg_p avg_pf avg_ux exec_time solve_ms notes; do
    [ "$label" = "mesh_label" ] && continue
    printf "%-8s %8s %-12s %8s %8s %8s %10s\n" \
        "$label" "$ncells" "$solver" "$avg_p" "$avg_pf" "$avg_ux" "$exec_time"
done < "$RESULTS_DIR/scaling_results.csv"

echo ""
echo "Results saved to $RESULTS_DIR/scaling_results.csv"
