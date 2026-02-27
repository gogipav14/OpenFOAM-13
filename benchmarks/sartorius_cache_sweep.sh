#!/bin/bash
# =============================================================================
# Sartorius 50L Cache Interval Sweep at 5M cells
# =============================================================================
# Tests mgCacheInterval = {10, 50, 100, 1000} at the 5M mesh level
# to quantify the AMG hierarchy rebuild cost.
#
# Uses the pre-generated 5M mesh from the scaling benchmark to save time.
#
# Usage (inside Docker container):
#   source /opt/OpenFOAM-13/etc/bashrc
#   bash /workspace/benchmarks/sartorius_cache_sweep.sh
# =============================================================================

set -eo pipefail

RESULTS_DIR="/workspace/benchmarks/sartorius_cache_results"
BASE_CASE="/workspace/benchmarks/sartorius_50L_benchmark"
NSTEPS=10       # 10 timesteps = 40 pressure solves
DT=0.005

# Mesh: 172x172x172 = ~5M cells
NX=172; NY=172; NZ=172

# Cache intervals to sweep
CACHE_INTERVALS=(50 100 1000)

# Momentum preconditioners to test at each interval
PRECONDITIONERS=("blockJacobi" "ILU")

# Append mode: keep existing results from previous partial run
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "Sartorius Cache Interval Sweep (5M cells)"
echo "============================================"
echo "Mesh: ${NX}x${NY}x${NZ} = $((NX*NY*NZ)) cells"
echo "Steps: $NSTEPS (dt=$DT)"
echo "Cache intervals: ${CACHE_INTERVALS[*]}"
echo "NOTE: Appending to existing CSV (ci10 results already collected)"
echo ""

# Write header only if CSV doesn't exist
if [ ! -f "$RESULTS_DIR/cache_sweep.csv" ]; then
    echo "cache_interval,preconditioner,avg_p_iters,avg_pFinal_iters,avg_Ux_iters,exec_time_s,avg_p_solve_ms,min_p_solve_ms,max_p_solve_ms,notes" \
        > "$RESULTS_DIR/cache_sweep.csv"
fi

# Remove stale entries for intervals we're about to re-run
for interval in "${CACHE_INTERVALS[@]}"; do
    sed -i "/^${interval},/d" "$RESULTS_DIR/cache_sweep.csv"
done

# Generate mesh once
meshdir="$RESULTS_DIR/_mesh_5M"
rm -rf "$meshdir"
cp -r "$BASE_CASE" "$meshdir"
find "$meshdir" -mindepth 1 -maxdepth 1 -name '[1-9]*' -type d -exec rm -rf {} + 2>/dev/null || true
find "$meshdir" -mindepth 1 -maxdepth 1 -name '0.*' -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf "$meshdir/postProcessing"

sed -i "s/^nx  [0-9]*/nx  $NX/" "$meshdir/system/blockMeshDict"
sed -i "s/^ny  [0-9]*/ny  $NY/" "$meshdir/system/blockMeshDict"
sed -i "s/^nz  [0-9]*/nz  $NZ/" "$meshdir/system/blockMeshDict"

echo "Generating 5M mesh (${NX}x${NY}x${NZ})..."
blockMesh -case "$meshdir" > /dev/null 2>&1
topoSet -case "$meshdir" > /dev/null 2>&1
rm -f "$meshdir/0/tracer"
echo "Mesh ready."
echo ""

write_fvsolution() {
    local casedir=$1
    local preconditioner=$2
    local cache_interval=$3

    local endTime
    endTime=$(python3 -c "print(f'{$NSTEPS * $DT:.6f}')")

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
            mgCacheInterval     $cache_interval;
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
            mgCacheInterval     $cache_interval;
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

    cat > "$casedir/system/controlDict" << EOFCD
FoamFile { format ascii; class dictionary; object controlDict; }
libs ("libOGL.so");
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

for interval in "${CACHE_INTERVALS[@]}"; do
    for precond in "${PRECONDITIONERS[@]}"; do
        label="${precond}_ci${interval}"
        casedir="$RESULTS_DIR/${label}"
        rm -rf "$casedir"
        cp -r "$meshdir" "$casedir"

        echo "--- $label (mgCacheInterval=$interval, momentum=$precond) ---"

        write_fvsolution "$casedir" "$precond" "$interval"

        echo "    Running ($NSTEPS steps)..."
        output=$(foamRun -case "$casedir" 2>&1 || true)
        echo "$output" > "$RESULTS_DIR/${label}.log"

        if echo "$output" | grep -qi "FOAM FATAL" 2>/dev/null; then
            echo "    *** FATAL ERROR ***"
            echo "$interval,$precond,FATAL,FATAL,FATAL,FATAL,FATAL,FATAL,FATAL,error" \
                >> "$RESULTS_DIR/cache_sweep.csv"
            continue
        fi

        # Extract pressure iterations
        avg_p=$( (echo "$output" | grep "OGLPCG.*Solving for p" || true) | \
            tail -n +2 | awk 'NR%2==1' | \
            awk -F'No Iterations ' '{s+=$2; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
        avg_pf=$( (echo "$output" | grep "OGLPCG.*Solving for p" || true) | \
            tail -n +2 | awk 'NR%2==0' | \
            awk -F'No Iterations ' '{s+=$2; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
        avg_ux=$( (echo "$output" | grep "OGLBiCGStab.*Solving for Ux" || true) | tail -n +2 | \
            awk -F'No Iterations ' '{s+=$2; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')

        # CG solve time stats (from pFinal debug=1)
        solve_times=$( (echo "$output" | grep "CG solve:" || true) | \
            awk -F'CG solve: ' '{print $2}' | awk '{print $1}' | tail -n +2)

        avg_ms=$(echo "$solve_times" | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
        min_ms=$(echo "$solve_times" | sort -n | head -1 | awk '{printf "%.1f", $1}')
        max_ms=$(echo "$solve_times" | sort -n | tail -1 | awk '{printf "%.1f", $1}')

        exec_time=$( (echo "$output" | grep "ExecutionTime" || true) | tail -1 | \
            awk -F'= ' '{print $2}' | awk '{print $1}')

        echo "    p=$avg_p  pFinal=$avg_pf  Ux=$avg_ux  time=${exec_time}s"
        echo "    CG solve: avg=${avg_ms}ms  min=${min_ms}ms  max=${max_ms}ms"

        echo "$interval,$precond,$avg_p,$avg_pf,$avg_ux,$exec_time,$avg_ms,$min_ms,$max_ms," \
            >> "$RESULTS_DIR/cache_sweep.csv"
    done
    echo ""
done

# =========================================================================
# BONUS: Clean CPU GAMG re-run at 5M for a reliable baseline
# =========================================================================
echo "============================================"
echo "CPU GAMG Baseline Re-run (5M cells)"
echo "============================================"

cpudir="$RESULTS_DIR/cpuGAMG_baseline"
rm -rf "$cpudir"
cp -r "$meshdir" "$cpudir"

endTime=$(python3 -c "print(f'{$NSTEPS * $DT:.6f}')")

cat > "$cpudir/system/fvSolution" << 'EOFSOL'
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

cat > "$cpudir/system/controlDict" << EOFCD
FoamFile { format ascii; class dictionary; object controlDict; }
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

echo "    Running CPU GAMG baseline ($NSTEPS steps)..."
output=$(foamRun -case "$cpudir" 2>&1 || true)
echo "$output" > "$RESULTS_DIR/cpuGAMG_baseline.log"

cpu_p=$( (echo "$output" | grep "GAMG.*Solving for p" || true) | \
    tail -n +2 | awk 'NR%2==1' | \
    awk -F'No Iterations ' '{s+=$2; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
cpu_pf=$( (echo "$output" | grep "GAMG.*Solving for p" || true) | \
    tail -n +2 | awk 'NR%2==0' | \
    awk -F'No Iterations ' '{s+=$2; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
cpu_ux=$( (echo "$output" | grep "smoothSolver.*Solving for Ux" || true) | tail -n +2 | \
    awk -F'No Iterations ' '{s+=$2; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
cpu_time=$( (echo "$output" | grep "ExecutionTime" || true) | tail -1 | \
    awk -F'= ' '{print $2}' | awk '{print $1}')

echo "    CPU GAMG: p=$cpu_p  pFinal=$cpu_pf  Ux=$cpu_ux  time=${cpu_time}s"
echo "0,cpuGAMG,$cpu_p,$cpu_pf,$cpu_ux,$cpu_time,N/A,N/A,N/A,baseline" \
    >> "$RESULTS_DIR/cache_sweep.csv"

# Clean up mesh
rm -rf "$meshdir"

echo ""
echo "============================================"
echo "Full Results Summary"
echo "============================================"
echo ""
printf "%-6s %-12s %8s %8s %10s %10s %10s %10s\n" \
    "CI" "Precond" "p_iters" "pF_iters" "Time(s)" "avg_ms" "min_ms" "max_ms"
printf "%s\n" "-------------------------------------------------------------------------------"

while IFS=',' read -r ci precond avg_p avg_pf avg_ux exec_time avg_ms min_ms max_ms notes; do
    [ "$ci" = "cache_interval" ] && continue
    printf "%-6s %-12s %8s %8s %10s %10s %10s %10s\n" \
        "$ci" "$precond" "$avg_p" "$avg_pf" "$exec_time" "$avg_ms" "$min_ms" "$max_ms"
done < "$RESULTS_DIR/cache_sweep.csv"

echo ""
echo "Results saved to $RESULTS_DIR/cache_sweep.csv"
