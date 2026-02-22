/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2025 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "OGLCylindricalSolver.H"
#include "OGLExecutor.H"
#include "FP32CastWrapper.H"
#include "addToRunTimeSelectionTable.H"
#include "polyMesh.H"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{
    defineTypeNameAndDebug(OGLCylindricalSolver, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<OGLCylindricalSolver>
        addOGLCylindricalSymMatrixConstructorToTable_;
}
}

std::map<Foam::word, Foam::OGL::OGLCylindricalSolver::CylCache>
    Foam::OGL::OGLCylindricalSolver::cache_;
std::mutex Foam::OGL::OGLCylindricalSolver::cacheMutex_;


// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

bool Foam::OGL::OGLCylindricalSolver::detectCylindricalMesh() const
{
    const auto& meshRef = matrix().mesh();
    const auto& mesh = dynamic_cast<const polyMesh&>(meshRef);

    // Get cell centres
    const vectorField& cc = mesh.cellCentres();

    // Determine which cells to process
    labelList targetCells;
    if (useZone_)
    {
        const auto& cellZones = mesh.cellZones();
        label zoneID = cellZones.findIndex(cylindricalZoneName_);
        if (zoneID < 0)
        {
            FatalErrorInFunction
                << "cellZone '" << cylindricalZoneName_ << "' not found"
                << abort(FatalError);
        }
        targetCells = cellZones[zoneID];
        sort(targetCells);
        zoneCells_ = targetCells;
    }
    else
    {
        // All cells
        label nCells = cc.size();
        targetCells.setSize(nCells);
        forAll(targetCells, i) { targetCells[i] = i; }
    }

    label nTarget = targetCells.size();
    if (nTarget == 0) return false;

    // Project cell centres to cylindrical coordinates
    // Axis: cylindricalAxis_ through cylindricalOrigin_
    vector axisDir = cylindricalAxis_ / mag(cylindricalAxis_);

    scalarList rVals(nTarget);
    scalarList thetaVals(nTarget);

    forAll(targetCells, ci)
    {
        vector rel = cc[targetCells[ci]] - cylindricalOrigin_;

        // Axial component
        scalar axial = rel & axisDir;

        // Radial vector (perpendicular to axis)
        vector radVec = rel - axial * axisDir;
        scalar r = mag(radVec);
        rVals[ci] = r;

        // Angle in the plane perpendicular to axis
        // Build orthogonal basis: e1 = arbitrary perp to axis, e2 = axis x e1
        // Use a stable choice for the reference direction
        scalar theta = Foam::atan2(radVec.y(), radVec.x());
        if (theta < 0) theta += 2.0 * constant::mathematical::pi;
        thetaVals[ci] = theta;
    }

    // Detect unique radii and angles by quantization.
    // Use a tolerance large enough to merge cells at the same radial index
    // that have slightly different radii due to arc-edge mesh interpolation
    // across different angular sectors (typically ~1e-4 relative deviation).
    scalar rMin = min(rVals);
    scalar rMax = max(rVals);
    scalar rRange = rMax - rMin;
    scalar tol = 1e-3 * rRange;
    if (tol < SMALL) tol = SMALL;

    // Sort unique radii
    std::set<scalar> uniqueR;
    forAll(rVals, i)
    {
        scalar r = rVals[i];
        // Find existing close value
        bool found = false;
        for (const auto& ur : uniqueR)
        {
            if (mag(r - ur) < tol) { found = true; break; }
        }
        if (!found) uniqueR.insert(r);
    }

    // Sort unique angles
    scalar thetaTol = 1e-3 * 2.0 * constant::mathematical::pi;
    std::set<scalar> uniqueTheta;
    forAll(thetaVals, i)
    {
        scalar t = thetaVals[i];
        bool found = false;
        for (const auto& ut : uniqueTheta)
        {
            if (mag(t - ut) < thetaTol) { found = true; break; }
        }
        if (!found) uniqueTheta.insert(t);
    }

    nr_ = uniqueR.size();
    ntheta_ = uniqueTheta.size();

    if (nr_ * ntheta_ != nTarget)
    {
        WarningInFunction
            << "Cylindrical detection: nr=" << nr_
            << " * ntheta=" << ntheta_
            << " = " << (nr_ * ntheta_)
            << " != " << nTarget << " cells"
            << endl;
        return false;
    }

    // Build sorted radius and angle arrays for indexing
    std::vector<scalar> sortedR(uniqueR.begin(), uniqueR.end());
    std::vector<scalar> sortedTheta(uniqueTheta.begin(), uniqueTheta.end());

    // Build permutation: OpenFOAM cell -> structured (i_r * ntheta + i_theta)
    permutation_.setSize(nTarget);
    structuredToFoam_.setSize(nTarget);

    forAll(targetCells, ci)
    {
        // Find radial index
        label ir = -1;
        for (label i = 0; i < nr_; i++)
        {
            if (mag(rVals[ci] - sortedR[i]) < tol)
            {
                ir = i;
                break;
            }
        }

        // Find angular index
        label itheta = -1;
        for (label i = 0; i < ntheta_; i++)
        {
            if (mag(thetaVals[ci] - sortedTheta[i]) < thetaTol)
            {
                itheta = i;
                break;
            }
        }

        if (ir < 0 || itheta < 0)
        {
            WarningInFunction
                << "Cell " << targetCells[ci]
                << " not matched to cylindrical grid"
                << endl;
            return false;
        }

        label structIdx = ir * ntheta_ + itheta;
        permutation_[ci] = structIdx;
        structuredToFoam_[structIdx] = targetCells[ci];
    }

    detected_ = true;

    Info<< "OGLCylindrical: Detected cylindrical mesh: "
        << nr_ << " x " << ntheta_
        << " = " << nTarget << " cells" << nl
        << "  radii: " << rMin << " to " << rMax << nl
        << "  axis: " << axisDir
        << " through " << cylindricalOrigin_ << endl;

    return true;
}


void Foam::OGL::OGLCylindricalSolver::extractCylindricalCoeffs() const
{
    const auto& addr = matrix_.lduAddr();
    const labelUList& owner = addr.lowerAddr();
    const labelUList& neighbour = addr.upperAddr();
    const scalarField& upper = matrix_.upper();

    label nTarget = nr_ * ntheta_;

    // Build cell-to-structured-index lookup
    // For zone mode: only zone cells are in the permutation
    boolList inTarget(matrix_.diag().size(), false);
    labelList cellToStructIdx(matrix_.diag().size(), -1);

    if (useZone_)
    {
        forAll(zoneCells_, ci)
        {
            inTarget[zoneCells_[ci]] = true;
            cellToStructIdx[zoneCells_[ci]] = permutation_[ci];
        }
    }
    else
    {
        forAll(permutation_, ci)
        {
            inTarget[ci] = true;
            cellToStructIdx[ci] = permutation_[ci];
        }
    }

    // Classify faces: radial (different ir, same itheta) or angular
    // (same ir, different itheta).
    //
    // For each internal face with both cells in the target set:
    //   struct_o = i_r * ntheta + i_theta
    //   struct_n = j_r * ntheta + j_theta
    //
    // Radial face: i_theta == j_theta, |i_r - j_r| == 1
    //   -> coupling coefficient goes to lower/upper at i_r
    // Angular face: i_r == j_r, |i_theta - j_theta| == 1 or ntheta-1
    //   -> coupling coefficient goes to thetaCoeff at i_r

    // Accumulate per-radius coupling values
    std::vector<double> sumRadialLower(nr_, 0.0);
    std::vector<double> sumRadialUpper(nr_, 0.0);
    std::vector<double> sumTheta(nr_, 0.0);
    std::vector<int> cntRadialLower(nr_, 0);
    std::vector<int> cntRadialUpper(nr_, 0);
    std::vector<int> cntTheta(nr_, 0);

    forAll(owner, facei)
    {
        label o = owner[facei];
        label n = neighbour[facei];
        if (!inTarget[o] || !inTarget[n]) continue;

        label so = cellToStructIdx[o];
        label sn = cellToStructIdx[n];
        if (so < 0 || sn < 0) continue;

        label ir_o = so / ntheta_;
        label it_o = so % ntheta_;
        label ir_n = sn / ntheta_;
        label it_n = sn % ntheta_;

        double val = Foam::mag(upper[facei]);

        if (it_o == it_n && Foam::mag(ir_n - ir_o) == 1)
        {
            // Radial face
            if (ir_n > ir_o)
            {
                // Upper coupling for cell at ir_o
                sumRadialUpper[ir_o] += val;
                cntRadialUpper[ir_o]++;
                // Lower coupling for cell at ir_n
                sumRadialLower[ir_n] += val;
                cntRadialLower[ir_n]++;
            }
            else
            {
                sumRadialUpper[ir_n] += val;
                cntRadialUpper[ir_n]++;
                sumRadialLower[ir_o] += val;
                cntRadialLower[ir_o]++;
            }
        }
        else if (ir_o == ir_n)
        {
            // Angular face (same radius)
            label diff = Foam::mag(it_n - it_o);
            if (diff == 1 || diff == ntheta_ - 1)
            {
                // Average into the shared radius
                sumTheta[ir_o] += val;
                cntTheta[ir_o]++;
            }
        }
    }

    // Compute mean coefficients per radius
    std::vector<double> lower(nr_, 0.0);
    std::vector<double> upperCoeff(nr_, 0.0);
    std::vector<double> thetaCoeff(nr_, 0.0);

    for (label i = 0; i < nr_; i++)
    {
        if (cntRadialLower[i] > 0)
            lower[i] = sumRadialLower[i] / cntRadialLower[i];
        if (cntRadialUpper[i] > 0)
            upperCoeff[i] = sumRadialUpper[i] / cntRadialUpper[i];
        if (cntTheta[i] > 0)
            thetaCoeff[i] = sumTheta[i] / cntTheta[i];
    }

    if (debug_ >= 1)
    {
        Info<< "OGLCylindrical: coupling coefficients:" << nl;
        for (label i = 0; i < nr_; i++)
        {
            Info<< "  r[" << i << "]: lower=" << lower[i]
                << " upper=" << upperCoeff[i]
                << " theta=" << thetaCoeff[i] << nl;
        }
        Info<< endl;
    }

    // Update the cylindrical preconditioner
    if (precisionPolicy_ == PrecisionPolicy::FP64)
    {
        cylF64_->setCoeffs(lower.data(), upperCoeff.data(), thetaCoeff.data());
    }
    else
    {
        cylF32_->setCoeffs(lower.data(), upperCoeff.data(), thetaCoeff.data());
    }
}


template<typename ValueType>
void Foam::OGL::OGLCylindricalSolver::applyCylPrecond
(
    scalarField& z,
    const scalarField& r,
    std::shared_ptr<CylFFTPreconditioner<ValueType>>& precond
) const
{
    auto exec = OGLExecutor::instance().executor();
    label nStructured = nr_ * ntheta_;

    // Gather residual into structured (r, theta) order
    scalarField structR(nStructured, 0.0);
    if (useZone_)
    {
        forAll(zoneCells_, ci)
        {
            structR[permutation_[ci]] = r[zoneCells_[ci]];
        }
    }
    else
    {
        forAll(permutation_, ci)
        {
            structR[permutation_[ci]] = r[ci];
        }
    }

    // Apply preconditioner via GPU
    scalarField structZ(nStructured, 0.0);
    {
        std::shared_ptr<gko::matrix::Dense<ValueType>> gb, gx;
        if constexpr (std::is_same<ValueType, float>::value)
        {
            gb = FP32CastWrapper::toGinkgoF32(exec, structR);
            gx = FP32CastWrapper::toGinkgoF32(exec, structZ);
        }
        else
        {
            gb = FP32CastWrapper::toGinkgoF64(exec, structR);
            gx = FP32CastWrapper::toGinkgoF64(exec, structZ);
        }

        precond->apply(gb.get(), gx.get());

        if constexpr (std::is_same<ValueType, float>::value)
        {
            FP32CastWrapper::fromGinkgoF32(gx.get(), structZ);
        }
        else
        {
            FP32CastWrapper::fromGinkgoF64(gx.get(), structZ);
        }
    }

    if (debug_ >= 2)
    {
        scalar rNorm = 0, zNorm = 0, rz = 0;
        forAll(structR, i)
        {
            rNorm += Foam::mag(structR[i]);
            zNorm += Foam::mag(structZ[i]);
            rz += structR[i] * structZ[i];
        }
        Info<< "  CylPrecond: |structR|=" << rNorm
            << " |structZ|=" << zNorm
            << " <r,z>=" << rz << endl;
    }

    // Scatter back to OpenFOAM cell order
    if (useZone_)
    {
        // Jacobi for non-zone cells
        const scalarField& diag = matrix_.diag();
        forAll(z, i) { z[i] = r[i] / diag[i]; }

        // Overwrite zone cells with cylindrical preconditioner result
        forAll(zoneCells_, ci)
        {
            z[zoneCells_[ci]] = structZ[permutation_[ci]];
        }
    }
    else
    {
        forAll(permutation_, ci)
        {
            z[ci] = structZ[permutation_[ci]];
        }
    }
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

Foam::label Foam::OGL::OGLCylindricalSolver::solveFP32
(
    scalarField& psi,
    const scalarField& source,
    const scalar tolerance
) const
{
    // Not used — solve() implements its own PCG loop
    return 0;
}


Foam::label Foam::OGL::OGLCylindricalSolver::solveFP64
(
    scalarField& psi,
    const scalarField& source,
    const scalar tolerance
) const
{
    return 0;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::OGL::OGLCylindricalSolver::OGLCylindricalSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    OGLSolverBase
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    cylindricalOrigin_(vector::zero),
    cylindricalAxis_(vector(0, 0, 1)),
    cylindricalZoneName_(),
    useZone_(false),
    nr_(0),
    ntheta_(0),
    detected_(false),
    cylF32_(nullptr),
    cylF64_(nullptr),
    coeffsInitialized_(false)
{
    // Default to more refinement iterations
    if (controlDict_.found("OGLCoeffs"))
    {
        const dictionary& d = controlDict_.subDict("OGLCoeffs");
        if (!d.found("maxRefineIters"))
        {
            maxRefineIters_ = 10;
        }
    }
    else
    {
        maxRefineIters_ = 10;
    }

    // Read cylindrical parameters
    if (controlDict_.found("OGLCoeffs"))
    {
        const dictionary& d = controlDict_.subDict("OGLCoeffs");

        if (d.found("cylindricalOrigin"))
        {
            cylindricalOrigin_ =
                d.lookup<vector>("cylindricalOrigin");
        }
        if (d.found("cylindricalAxis"))
        {
            cylindricalAxis_ =
                d.lookup<vector>("cylindricalAxis");
        }
        if (d.found("cylindricalZone"))
        {
            cylindricalZoneName_ =
                d.lookup<word>("cylindricalZone");
            useZone_ = true;
        }
    }

    // Restore cached state
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto it = cache_.find(fieldName);
        if (it != cache_.end())
        {
            cylF32_ = it->second.cylF32;
            cylF64_ = it->second.cylF64;
            coeffsInitialized_ = it->second.coeffsInitialized;
        }
    }

    Info<< "OGLCylindrical: Created for field " << fieldName << nl
        << "  origin: " << cylindricalOrigin_ << nl
        << "  axis: " << cylindricalAxis_ << nl
        << "  zone: "
        << (useZone_ ? cylindricalZoneName_ : word("none (full mesh)"))
        << nl
        << "  precisionPolicy: "
        << (precisionPolicy_ == PrecisionPolicy::FP64 ? "FP64" :
            precisionPolicy_ == PrecisionPolicy::FP32 ? "FP32" : "MIXED")
        << endl;
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

Foam::solverPerformance Foam::OGL::OGLCylindricalSolver::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    word solverName =
        type() + ":"
      + (precisionPolicy_ == PrecisionPolicy::FP64 ? "FP64" :
         precisionPolicy_ == PrecisionPolicy::FP32 ? "FP32" : "MIXED");

    solverPerformance solverPerf(solverName, fieldName_);

    const label nCells = psi.size();

    // --- Compute A*psi for normFactor + initial residual
    scalarField wA(nCells);
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    scalarField tmpField(nCells);
    scalar normFactor = this->normFactor(psi, source, wA, tmpField);

    scalarField rField(nCells);
    forAll(rField, i) { rField[i] = source[i] - wA[i]; }

    scalar initResidual =
        gSumMag(rField, matrix().mesh().comm()) / normFactor;
    solverPerf.initialResidual() = initResidual;
    solverPerf.finalResidual() = initResidual;

    if
    (
        minIter_ <= 0
     && solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        solverPerf.nIterations() = 0;
        return solverPerf;
    }

    // --- Detect cylindrical mesh (once)
    if (!detected_)
    {
        if (!detectCylindricalMesh())
        {
            FatalErrorInFunction
                << "OGLCylindrical: Failed to detect cylindrical mesh "
                << "structure. Check cylindricalOrigin and cylindricalAxis."
                << abort(FatalError);
        }
    }

    // --- Create preconditioner (once) and extract coefficients (every call)
    {
        auto exec = OGLExecutor::instance().executor();
        auto n = static_cast<gko::size_type>(nr_ * ntheta_);

        if (precisionPolicy_ == PrecisionPolicy::FP64)
        {
            if (!cylF64_)
            {
                cylF64_ = gko::share(
                    CylFFTPreconditioner<double>::create(
                        exec, gko::dim<2>{n, n},
                        nr_, ntheta_
                    )
                );
                Info<< "OGLCylindrical: preconditioner created "
                    << nr_ << "x" << ntheta_ << endl;
            }
        }
        else
        {
            if (!cylF32_)
            {
                cylF32_ = gko::share(
                    CylFFTPreconditioner<float>::create(
                        exec, gko::dim<2>{n, n},
                        nr_, ntheta_
                    )
                );
                Info<< "OGLCylindrical: preconditioner created "
                    << nr_ << "x" << ntheta_ << endl;
            }
        }

        // Re-extract coefficients every call to track rAU changes.
        // Thomas pre-factorization is O(nr * nModes) — negligible cost.
        extractCylindricalCoeffs();
        coeffsInitialized_ = true;
    }

    // --- Preconditioned CG (same as OGLSpectralSolver)
    scalarField zField(nCells, 0.0);
    scalarField pField(nCells, 0.0);
    scalarField qField(nCells);

    // z = M^{-1} * r
    if (precisionPolicy_ == PrecisionPolicy::FP64)
    {
        applyCylPrecond<double>(zField, rField, cylF64_);
    }
    else
    {
        applyCylPrecond<float>(zField, rField, cylF32_);
    }

    forAll(pField, i) { pField[i] = zField[i]; }

    scalar rz = gSumProd(rField, zField, matrix().mesh().comm());

    label totalIters = 0;

    do
    {
        matrix_.Amul(qField, pField, interfaceBouCoeffs_, interfaces_, cmpt);

        scalar pq = gSumProd(pField, qField, matrix().mesh().comm());

        if (solverPerf.checkSingularity(mag(pq) / normFactor))
        {
            break;
        }

        scalar alpha = rz / pq;

        forAll(psi, i)
        {
            psi[i] += alpha * pField[i];
            rField[i] -= alpha * qField[i];
        }

        totalIters++;

        solverPerf.finalResidual() =
            gSumMag(rField, matrix().mesh().comm()) / normFactor;

        if (debug_ >= 1)
        {
            Info<< "OGLCylindrical: PCG Iter " << totalIters
                << ", residual = " << solverPerf.finalResidual() << endl;
        }

        // z = M^{-1} * r
        forAll(zField, i) { zField[i] = 0.0; }
        if (precisionPolicy_ == PrecisionPolicy::FP64)
        {
            applyCylPrecond<double>(zField, rField, cylF64_);
        }
        else
        {
            applyCylPrecond<float>(zField, rField, cylF32_);
        }

        scalar rzNew = gSumProd(rField, zField, matrix().mesh().comm());
        scalar beta = rzNew / (rz + VSMALL);
        rz = rzNew;

        forAll(pField, i)
        {
            pField[i] = zField[i] + beta * pField[i];
        }

    } while
    (
        (
            totalIters < maxRefineIters_
        && !solverPerf.checkConvergence(tolerance_, relTol_)
        )
     || totalIters < minIter_
    );

    solverPerf.nIterations() = totalIters;

    // Persist cache
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto& entry = cache_[fieldName_];
        entry.cylF32 = cylF32_;
        entry.cylF64 = cylF64_;
        entry.coeffsInitialized = coeffsInitialized_;
    }

    return solverPerf;
}


// ************************************************************************* //
