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
    // Tolerance must be smaller than the angular cell spacing.
    // For N angular cells, spacing = 2*pi/N. Use 1e-6*2*pi ≈ 6e-6 rad
    // which supports up to ~1M angular cells.
    scalar thetaTol = 1e-6 * 2.0 * constant::mathematical::pi;
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

    // --- Sector DCT mode: use setCoeffsSector and skip Woodbury ---
    if (useSectorDCT_ && sectorsDetected_ && nSectors_ > 0)
    {
        if (precisionPolicy_ == PrecisionPolicy::FP64)
        {
            cylF64_->setCoeffsSector(
                lower.data(), upperCoeff.data(), thetaCoeff.data());
        }
        else
        {
            cylF32_->setCoeffsSector(
                lower.data(), upperCoeff.data(), thetaCoeff.data());
        }

        capK_ = 0;
        capValid_ = false;

        if (debug_ >= 1)
        {
            Info<< "OGLCylindrical: sector DCT coefficients set" << endl;
        }
        return;
    }

    // Update the cylindrical preconditioner (DFT mode)
    if (precisionPolicy_ == PrecisionPolicy::FP64)
    {
        cylF64_->setCoeffs(lower.data(), upperCoeff.data(), thetaCoeff.data());
    }
    else
    {
        cylF32_->setCoeffs(lower.data(), upperCoeff.data(), thetaCoeff.data());
    }

    // --- Identify defect angular faces for Woodbury correction ---
    // Build a lookup of which angular face pairs actually exist as internal
    // faces. An angular face pair is (i_r, j) <-> (i_r, (j+1) % ntheta).
    // A complete annular mesh has nr * ntheta such pairs. Missing pairs
    // (blade walls) need Woodbury correction.

    // Map: structured angular pair index -> actual coupling value
    //   pair index = i_r * ntheta + j  for pair (i_r,j)-(i_r,(j+1)%ntheta)
    label nPairs = nr_ * ntheta_;
    std::vector<double> actualTheta(nPairs, 0.0);
    std::vector<bool> pairExists(nPairs, false);

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

        if (ir_o == ir_n)
        {
            label diff = Foam::mag(it_n - it_o);
            if (diff == 1 || diff == ntheta_ - 1)
            {
                // Angular face: identify the pair index
                // Convention: pair (ir, min_theta) for adjacent faces,
                // pair (ir, ntheta-1) for the wrap-around face
                label jLow, jHigh;
                if (diff == 1)
                {
                    jLow = Foam::min(it_o, it_n);
                }
                else // wrap-around: ntheta-1 <-> 0
                {
                    jLow = ntheta_ - 1;
                }
                label pairIdx = ir_o * ntheta_ + jLow;
                actualTheta[pairIdx] = Foam::mag(upper[facei]);
                pairExists[pairIdx] = true;
            }
        }
    }

    // Identify defect faces: angular face pairs where NO internal face
    // exists (blade walls). These are the only locations where the
    // periodicity assumption of the DFT is truly broken. Natural
    // coefficient variation (due to variable rAU) is handled by the PCG
    // iterations and should NOT be included in the Woodbury correction
    // (would make k ≈ N, defeating the purpose).
    capCellA_.clear();
    capCellB_.clear();
    capCoeffs_.clear();

    for (label ir = 0; ir < nr_; ir++)
    {
        double meanC = thetaCoeff[ir];
        if (meanC < SMALL) continue;

        for (label j = 0; j < ntheta_; j++)
        {
            label pairIdx = ir * ntheta_ + j;

            if (!pairExists[pairIdx])
            {
                // Missing face — blade wall. M assumes coupling = meanC
                // but the actual coupling is 0.
                label jNext = (j + 1) % ntheta_;
                label sA = ir * ntheta_ + j;
                label sB = ir * ntheta_ + jNext;

                capCellA_.append(sA);
                capCellB_.append(sB);
                // Perturbation: A has 0, M assumes meanC, so ΔA = -meanC
                // In Woodbury form: ΔA = U·C_diag·U^T with C_diag = meanC
                capCoeffs_.append(meanC);
            }
        }
    }

    capK_ = capCellA_.size();

    if (debug_ >= 1)
    {
        Info<< "OGLCylindrical: Woodbury correction: k=" << capK_
            << " defect angular faces" << endl;
    }
}


void Foam::OGL::OGLCylindricalSolver::detectSectors() const
{
    if (sectorsDetected_) return;

    // Scan internal faces to find missing angular pairs (blade walls)
    const auto& addr = matrix_.lduAddr();
    const labelUList& owner = addr.lowerAddr();
    const labelUList& neighbour = addr.upperAddr();

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

    // Check which angular face pairs exist
    std::vector<bool> pairExists(nr_ * ntheta_, false);

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

        if (ir_o == ir_n)
        {
            label diff = Foam::mag(it_n - it_o);
            if (diff == 1 || diff == ntheta_ - 1)
            {
                label jLow = (diff == 1)
                    ? Foam::min(it_o, it_n)
                    : ntheta_ - 1;
                pairExists[ir_o * ntheta_ + jLow] = true;
            }
        }
    }

    // Identify blade positions from first radial ring
    labelList bladeThetas;
    for (label j = 0; j < ntheta_; j++)
    {
        if (!pairExists[j])  // ir=0 ring
        {
            bladeThetas.append(j);
        }
    }

    nSectors_ = bladeThetas.size();

    if (nSectors_ == 0)
    {
        Info<< "OGLCylindrical: No blade walls detected, "
            << "sector DCT not applicable. Using DFT mode." << endl;
        sectorsDetected_ = true;
        return;
    }

    // Verify uniform sector sizes
    nthetaSector_ = ntheta_ / nSectors_;
    if (nthetaSector_ * nSectors_ != ntheta_)
    {
        FatalErrorInFunction
            << "Non-uniform sector sizes: ntheta=" << ntheta_
            << " not divisible by nSectors=" << nSectors_
            << abort(FatalError);
    }

    // Build sector start indices
    // bladeThetas[s] = last theta index of sector s (pair after it is missing)
    // Sector s starts at (bladeThetas[s] + 1) % ntheta_
    sort(bladeThetas);
    sectorStarts_.setSize(nSectors_);
    for (label s = 0; s < nSectors_; s++)
    {
        sectorStarts_[s] = (bladeThetas[s] + 1) % ntheta_;
    }

    // Verify all sectors have equal size
    for (label s = 0; s < nSectors_; s++)
    {
        label sNext = (s + 1) % nSectors_;
        label size;
        if (sectorStarts_[sNext] > sectorStarts_[s])
        {
            size = sectorStarts_[sNext] - sectorStarts_[s];
        }
        else
        {
            size = ntheta_ - sectorStarts_[s] + sectorStarts_[sNext];
        }
        if (size != nthetaSector_)
        {
            FatalErrorInFunction
                << "Non-uniform sector: sector " << s
                << " has " << size << " cells, expected " << nthetaSector_
                << abort(FatalError);
        }
    }

    // Build structToSector/sectorToStruct permutations
    // Structured: data[ir * ntheta + itheta]
    // Sector-ordered: data[(s * nr + ir) * nthetaSector + itheta_local]
    label nStruct = nr_ * ntheta_;
    structToSector_.setSize(nStruct);
    sectorToStruct_.setSize(nStruct);

    for (label s = 0; s < nSectors_; s++)
    {
        for (label ir = 0; ir < nr_; ir++)
        {
            for (label jLocal = 0; jLocal < nthetaSector_; jLocal++)
            {
                label jGlobal = (sectorStarts_[s] + jLocal) % ntheta_;
                label structIdx = ir * ntheta_ + jGlobal;
                label sectorIdx = (s * nr_ + ir) * nthetaSector_ + jLocal;

                structToSector_[structIdx] = sectorIdx;
                sectorToStruct_[sectorIdx] = structIdx;
            }
        }
    }

    sectorsDetected_ = true;

    Info<< "OGLCylindrical: Detected " << nSectors_ << " sectors"
        << " x " << nthetaSector_ << " theta cells each" << nl
        << "  sector starts:";
    forAll(sectorStarts_, s)
    {
        Info<< " " << sectorStarts_[s];
    }
    Info<< endl;
}


void Foam::OGL::OGLCylindricalSolver::applySpectralColumn
(
    scalarField& z,
    const scalarField& r
) const
{
    // Apply spectral preconditioner to structured-order vectors (CPU-side).
    // Must use the SAME precision as applyCylPrecond to maintain symmetry
    // of the Woodbury-corrected preconditioner (required for PCG).
    auto exec = OGLExecutor::instance().executor();

    if (precisionPolicy_ == PrecisionPolicy::FP64)
    {
        auto gb = FP32CastWrapper::toGinkgoF64(exec, r);
        auto gx = FP32CastWrapper::toGinkgoF64(exec, z);
        cylF64_->apply(gb.get(), gx.get());
        FP32CastWrapper::fromGinkgoF64(gx.get(), z);
    }
    else
    {
        auto gb = FP32CastWrapper::toGinkgoF32(exec, r);
        auto gx = FP32CastWrapper::toGinkgoF32(exec, z);
        cylF32_->apply(gb.get(), gx.get());
        FP32CastWrapper::fromGinkgoF32(gx.get(), z);
    }
}


void Foam::OGL::OGLCylindricalSolver::buildCapacitanceMatrix() const
{
    if (capK_ == 0)
    {
        capValid_ = false;
        return;
    }

    label nStruct = nr_ * ntheta_;

    // Build W = M^{-1} U (k spectral solves on unit vectors)
    // U has columns u_f where u_f[cellA_f] = +1, u_f[cellB_f] = -1
    capW_.setSize(capK_ * nStruct, 0.0);

    for (label f = 0; f < capK_; f++)
    {
        scalarField colR(nStruct, 0.0);
        scalarField colZ(nStruct, 0.0);

        // u_f = e_{cellA} - e_{cellB}
        colR[capCellA_[f]] = 1.0;
        colR[capCellB_[f]] = -1.0;

        applySpectralColumn(colZ, colR);

        // Store column f of W
        forAll(colZ, i)
        {
            capW_[f * nStruct + i] = colZ[i];
        }
    }

    // Build capacitance matrix S = C_diag^{-1} + U^T W  (k x k)
    //
    // The ldu matrix stores the negative-definite Laplacian (positive
    // off-diagonal, negative diagonal). In this convention, removing a
    // blade-wall angular coupling is a rank-1 ADDITION to the operator:
    //   A_stored = M_stored + U * C_diag * U^T
    //
    // Woodbury: (M + UCU^T)^{-1} = M^{-1} - M^{-1}*U*S^{-1}*U^T*M^{-1}
    //   where S = C_diag^{-1} + U^T*M^{-1}*U = C_diag^{-1} + U^T*W
    scalarList S(capK_ * capK_, 0.0);

    for (label i = 0; i < capK_; i++)
    {
        for (label j = 0; j < capK_; j++)
        {
            // (U^T W)_{ij} = u_i^T w_j
            //              = w_j[cellA_i] - w_j[cellB_i]
            double utw = capW_[j * nStruct + capCellA_[i]]
                       - capW_[j * nStruct + capCellB_[i]];

            S[i * capK_ + j] = utw;
        }
        // Add C_diag^{-1} on diagonal
        S[i * capK_ + i] += 1.0 / capCoeffs_[i];
    }

    // Factor S using dense LU (Gaussian elimination with partial pivoting)
    // Store S^{-1} directly for small k
    // For simplicity, compute S^{-1} by solving S * X = I column by column
    capSinv_.setSize(capK_ * capK_, 0.0);

    // Copy S for in-place factorization
    scalarList Scopy(S);
    labelList piv(capK_);

    // LU factorisation with partial pivoting
    for (label i = 0; i < capK_; i++)
    {
        // Find pivot
        label maxRow = i;
        double maxVal = Foam::mag(Scopy[i * capK_ + i]);
        for (label r = i + 1; r < capK_; r++)
        {
            double v = Foam::mag(Scopy[r * capK_ + i]);
            if (v > maxVal) { maxVal = v; maxRow = r; }
        }
        piv[i] = maxRow;

        // Swap rows
        if (maxRow != i)
        {
            for (label c = 0; c < capK_; c++)
            {
                std::swap(Scopy[i * capK_ + c], Scopy[maxRow * capK_ + c]);
            }
        }

        if (Foam::mag(Scopy[i * capK_ + i]) < VSMALL)
        {
            WarningInFunction
                << "Capacitance matrix is singular at row " << i
                << endl;
            capValid_ = false;
            return;
        }

        // Elimination
        for (label r = i + 1; r < capK_; r++)
        {
            double factor = Scopy[r * capK_ + i] / Scopy[i * capK_ + i];
            Scopy[r * capK_ + i] = factor;  // Store L
            for (label c = i + 1; c < capK_; c++)
            {
                Scopy[r * capK_ + c] -= factor * Scopy[i * capK_ + c];
            }
        }
    }

    // Solve S * X = I column by column to get S^{-1}
    for (label col = 0; col < capK_; col++)
    {
        scalarList rhs(capK_, 0.0);
        rhs[col] = 1.0;

        // Apply row permutation
        for (label i = 0; i < capK_; i++)
        {
            if (piv[i] != i)
            {
                std::swap(rhs[i], rhs[piv[i]]);
            }
        }

        // Forward substitution (L y = rhs)
        for (label i = 1; i < capK_; i++)
        {
            for (label j = 0; j < i; j++)
            {
                rhs[i] -= Scopy[i * capK_ + j] * rhs[j];
            }
        }

        // Backward substitution (U x = y)
        for (label i = capK_ - 1; i >= 0; i--)
        {
            for (label j = i + 1; j < capK_; j++)
            {
                rhs[i] -= Scopy[i * capK_ + j] * rhs[j];
            }
            rhs[i] /= Scopy[i * capK_ + i];
        }

        // Store column of S^{-1}
        for (label i = 0; i < capK_; i++)
        {
            capSinv_[i * capK_ + col] = rhs[i];
        }
    }

    capValid_ = true;

    Info<< "OGLCylindrical: Capacitance matrix built: k=" << capK_
        << " (rank " << capK_ << " Woodbury correction)" << endl;
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

    // Apply preconditioner: single FFT+Thomas apply + Woodbury correction
    scalarField structZ(nStructured, 0.0);

    {
        // --- Single FFT+Thomas apply ---
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

        // --- Woodbury correction: z = z0 - W S^{-1} U^T z0 ---
        if (capValid_ && capK_ > 0)
        {
            // Step 1: t = U^T z0 (k entries)
            scalarList t(capK_, 0.0);
            for (label f = 0; f < capK_; f++)
            {
                t[f] = structZ[capCellA_[f]] - structZ[capCellB_[f]];
            }

            // Step 2: s = S^{-1} t (k x k dense matvec)
            scalarList s(capK_, 0.0);
            for (label i = 0; i < capK_; i++)
            {
                for (label j = 0; j < capK_; j++)
                {
                    s[i] += capSinv_[i * capK_ + j] * t[j];
                }
            }

            // Step 3: correction = W s (N entries)
            for (label f = 0; f < capK_; f++)
            {
                for (label i = 0; i < nStructured; i++)
                {
                    structZ[i] -= capW_[f * nStructured + i] * s[f];
                }
            }
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
        if (fallbackType_ == DIC)
        {
            // DIC forward/backward sweep on all cells
            const labelUList& uAddr = matrix_.lduAddr().upperAddr();
            const labelUList& lAddr = matrix_.lduAddr().lowerAddr();
            const scalarField& upper = matrix_.upper();
            const label nFaces = upper.size();

            // Forward sweep: w = rD * r, then forward elimination
            forAll(z, i) { z[i] = rDDIC_[i] * r[i]; }
            for (label face = 0; face < nFaces; face++)
            {
                z[uAddr[face]] -=
                    rDDIC_[uAddr[face]] * upper[face] * z[lAddr[face]];
            }
            // Backward sweep
            for (label face = nFaces - 1; face >= 0; face--)
            {
                z[lAddr[face]] -=
                    rDDIC_[lAddr[face]] * upper[face] * z[uAddr[face]];
            }
        }
        else
        {
            // Jacobi for non-zone cells
            const scalarField& diag = matrix_.diag();
            forAll(z, i) { z[i] = r[i] / diag[i]; }
        }

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


template<typename ValueType>
void Foam::OGL::OGLCylindricalSolver::applyCylPrecondSector
(
    scalarField& z,
    const scalarField& r,
    std::shared_ptr<CylFFTPreconditioner<ValueType>>& precond
) const
{
    auto exec = OGLExecutor::instance().executor();
    label nStructured = nr_ * ntheta_;

    // Gather residual to sector-ordered layout
    scalarField sectorR(nStructured, 0.0);
    if (useZone_)
    {
        forAll(zoneCells_, ci)
        {
            label structIdx = permutation_[ci];
            sectorR[structToSector_[structIdx]] = r[zoneCells_[ci]];
        }
    }
    else
    {
        forAll(permutation_, ci)
        {
            label structIdx = permutation_[ci];
            sectorR[structToSector_[structIdx]] = r[ci];
        }
    }

    // Apply sector DCT+Thomas preconditioner
    scalarField sectorZ(nStructured, 0.0);
    {
        std::shared_ptr<gko::matrix::Dense<ValueType>> gb, gx;
        if constexpr (std::is_same<ValueType, float>::value)
        {
            gb = FP32CastWrapper::toGinkgoF32(exec, sectorR);
            gx = FP32CastWrapper::toGinkgoF32(exec, sectorZ);
        }
        else
        {
            gb = FP32CastWrapper::toGinkgoF64(exec, sectorR);
            gx = FP32CastWrapper::toGinkgoF64(exec, sectorZ);
        }

        precond->apply(gb.get(), gx.get());

        if constexpr (std::is_same<ValueType, float>::value)
        {
            FP32CastWrapper::fromGinkgoF32(gx.get(), sectorZ);
        }
        else
        {
            FP32CastWrapper::fromGinkgoF64(gx.get(), sectorZ);
        }
    }

    if (debug_ >= 2)
    {
        scalar rNorm = 0, zNorm = 0, rz = 0;
        forAll(sectorR, i)
        {
            rNorm += Foam::mag(sectorR[i]);
            zNorm += Foam::mag(sectorZ[i]);
            rz += sectorR[i] * sectorZ[i];
        }
        Info<< "  CylPrecondSector: |sectorR|=" << rNorm
            << " |sectorZ|=" << zNorm
            << " <r,z>=" << rz << endl;
    }

    // Scatter back to OpenFOAM cell order
    if (useZone_)
    {
        // Apply fallback preconditioner to all cells first
        if (fallbackType_ == DIC)
        {
            const labelUList& uAddr = matrix_.lduAddr().upperAddr();
            const labelUList& lAddr = matrix_.lduAddr().lowerAddr();
            const scalarField& upper = matrix_.upper();
            const label nFaces = upper.size();

            forAll(z, i) { z[i] = rDDIC_[i] * r[i]; }
            for (label face = 0; face < nFaces; face++)
            {
                z[uAddr[face]] -=
                    rDDIC_[uAddr[face]] * upper[face] * z[lAddr[face]];
            }
            for (label face = nFaces - 1; face >= 0; face--)
            {
                z[lAddr[face]] -=
                    rDDIC_[lAddr[face]] * upper[face] * z[uAddr[face]];
            }
        }
        else
        {
            const scalarField& diag = matrix_.diag();
            forAll(z, i) { z[i] = r[i] / diag[i]; }
        }

        // Overwrite zone cells with sector preconditioner result
        forAll(zoneCells_, ci)
        {
            label structIdx = permutation_[ci];
            z[zoneCells_[ci]] = sectorZ[structToSector_[structIdx]];
        }
    }
    else
    {
        forAll(permutation_, ci)
        {
            label structIdx = permutation_[ci];
            z[ci] = sectorZ[structToSector_[structIdx]];
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
    fallbackType_(JACOBI),
    nr_(0),
    ntheta_(0),
    detected_(false),
    cylF32_(nullptr),
    cylF64_(nullptr),
    coeffsInitialized_(false),
    capK_(0),
    capValid_(false),
    useSectorDCT_(false),
    nSectors_(0),
    nthetaSector_(0),
    sectorsDetected_(false)
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
        if (d.found("fallbackPreconditioner"))
        {
            word fb = d.lookup<word>("fallbackPreconditioner");
            if (fb == "DIC")
            {
                fallbackType_ = DIC;
            }
            else if (fb == "Jacobi" || fb == "jacobi")
            {
                fallbackType_ = JACOBI;
            }
            else
            {
                WarningInFunction
                    << "Unknown fallbackPreconditioner '" << fb
                    << "'. Using Jacobi." << endl;
            }
        }
        if (d.found("useSectorDCT"))
        {
            useSectorDCT_ = d.lookup<bool>("useSectorDCT");
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
        << "  fallback: "
        << (fallbackType_ == DIC ? "DIC" : "Jacobi") << nl
        << "  precisionPolicy: "
        << (precisionPolicy_ == PrecisionPolicy::FP64 ? "FP64" :
            precisionPolicy_ == PrecisionPolicy::FP32 ? "FP32" : "MIXED")
        << nl
        << "  sectorDCT: " << (useSectorDCT_ ? "on" : "off")
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

    // --- Detect sector boundaries (once, when useSectorDCT is enabled)
    if (useSectorDCT_ && !sectorsDetected_)
    {
        detectSectors();
    }

    bool sectorMode = useSectorDCT_ && sectorsDetected_ && nSectors_ > 0;

    // --- Create preconditioner (once) and extract coefficients (every call)
    {
        auto exec = OGLExecutor::instance().executor();
        auto n = static_cast<gko::size_type>(nr_ * ntheta_);

        if (precisionPolicy_ == PrecisionPolicy::FP64)
        {
            if (!cylF64_)
            {
                if (sectorMode)
                {
                    cylF64_ = gko::share(
                        CylFFTPreconditioner<double>::create(
                            exec, gko::dim<2>{n, n},
                            nr_, nSectors_, nthetaSector_
                        )
                    );
                    Info<< "OGLCylindrical: sector DCT preconditioner "
                        << nr_ << "x" << nSectors_ << "x" << nthetaSector_
                        << endl;
                }
                else
                {
                    cylF64_ = gko::share(
                        CylFFTPreconditioner<double>::create(
                            exec, gko::dim<2>{n, n},
                            nr_, ntheta_
                        )
                    );
                    Info<< "OGLCylindrical: DFT preconditioner "
                        << nr_ << "x" << ntheta_ << endl;
                }
            }
        }
        else
        {
            if (!cylF32_)
            {
                if (sectorMode)
                {
                    cylF32_ = gko::share(
                        CylFFTPreconditioner<float>::create(
                            exec, gko::dim<2>{n, n},
                            nr_, nSectors_, nthetaSector_
                        )
                    );
                    Info<< "OGLCylindrical: sector DCT preconditioner "
                        << nr_ << "x" << nSectors_ << "x" << nthetaSector_
                        << endl;
                }
                else
                {
                    cylF32_ = gko::share(
                        CylFFTPreconditioner<float>::create(
                            exec, gko::dim<2>{n, n},
                            nr_, ntheta_
                        )
                    );
                    Info<< "OGLCylindrical: DFT preconditioner "
                        << nr_ << "x" << ntheta_ << endl;
                }
            }
        }

        // Re-extract coefficients every call to track rAU changes.
        // Thomas pre-factorization is O(nr * nModes) — negligible cost.
        extractCylindricalCoeffs();
        coeffsInitialized_ = true;

        // Build Woodbury capacitance matrix (DFT mode only)
        if (!sectorMode && capK_ > 0)
        {
            buildCapacitanceMatrix();
        }
    }

    // --- Compute DIC factorization for non-zone fallback
    if (useZone_ && fallbackType_ == DIC)
    {
        rDDIC_ = matrix_.diag();
        DICPreconditioner::calcReciprocalD(rDDIC_, matrix_);
    }

    // --- Preconditioned CG (same as OGLSpectralSolver)
    scalarField zField(nCells, 0.0);
    scalarField pField(nCells, 0.0);
    scalarField qField(nCells);

    // z = M^{-1} * r
    if (sectorMode)
    {
        if (precisionPolicy_ == PrecisionPolicy::FP64)
            applyCylPrecondSector<double>(zField, rField, cylF64_);
        else
            applyCylPrecondSector<float>(zField, rField, cylF32_);
    }
    else
    {
        if (precisionPolicy_ == PrecisionPolicy::FP64)
            applyCylPrecond<double>(zField, rField, cylF64_);
        else
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
        if (sectorMode)
        {
            if (precisionPolicy_ == PrecisionPolicy::FP64)
                applyCylPrecondSector<double>(zField, rField, cylF64_);
            else
                applyCylPrecondSector<float>(zField, rField, cylF32_);
        }
        else
        {
            if (precisionPolicy_ == PrecisionPolicy::FP64)
                applyCylPrecond<double>(zField, rField, cylF64_);
            else
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
