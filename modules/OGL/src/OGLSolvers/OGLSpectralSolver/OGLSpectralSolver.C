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

#include "OGLSpectralSolver.H"
#include "OGLExecutor.H"
#include "FP32CastWrapper.H"
#include "HaloKernels.h"
#include "addToRunTimeSelectionTable.H"
#include "polyMesh.H"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{
    defineTypeNameAndDebug(OGLSpectralSolver, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<OGLSpectralSolver>
        addOGLSpectralSymMatrixConstructorToTable_;
}
}

// Static cache members
std::map<Foam::word, Foam::OGL::OGLSpectralSolver::SpectralCache>
    Foam::OGL::OGLSpectralSolver::cache_;
std::mutex Foam::OGL::OGLSpectralSolver::cacheMutex_;


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<typename ValueType>
void Foam::OGL::OGLSpectralSolver::ensureOperator
(
    std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op
) const
{
    auto exec = OGLExecutor::instance().executor();

    if (op)
    {
        // Reuse cached operator — update pointers and invalidate values
        op->updatePointers
        (
            matrix_,
            interfaceBouCoeffs_,
            interfaceIntCoeffs_,
            interfaces_,
            0  // cmpt
        );
        op->invalidateValues();
    }
    else
    {
        // First call — create operator from scratch
        op = std::make_shared<FoamGinkgoLinOp<ValueType>>
        (
            exec,
            matrix_,
            interfaceBouCoeffs_,
            interfaceIntCoeffs_,
            interfaces_,
            0,      // cmpt
            true,   // includeInterfaces
            cacheStructure_,
            cacheValues_
        );
    }
}


template<typename ValueType>
void Foam::OGL::OGLSpectralSolver::initFFTSolver
(
    std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op,
    std::shared_ptr<FFTPreconditioner<ValueType>>& fftSolver
) const
{
    auto exec = OGLExecutor::instance().executor();

    // Create FFT solver if not yet cached
    if (!fftSolver)
    {
        auto nCells = static_cast<gko::size_type>(
            fftDimensions_.x() * fftDimensions_.y() * fftDimensions_.z()
        );

        fftSolver = gko::share(
            FFTPreconditioner<ValueType>::create(
                exec,
                gko::dim<2>{nCells, nCells},
                fftDimensions_.x(),
                fftDimensions_.y(),
                fftDimensions_.z(),
                double(meshSpacing_.x()),
                double(meshSpacing_.y()),
                double(meshSpacing_.z())
            )
        );
    }

    // In zone mode, extract coupling coefficients directly from lduMatrix
    // face addressing (not from CSR). The zone cells have non-contiguous
    // global indices, so the CSR neighbor detection would need full-mesh
    // strides rather than zone dimensions. The ldu approach is simpler.
    if (useZone_)
    {
        const auto& addr = matrix_.lduAddr();
        const labelUList& owner = addr.lowerAddr();
        const labelUList& neighbour = addr.upperAddr();
        const scalarField& upper = matrix_.upper();

        // Detect full-mesh strides from cell count + zone dimensions.
        // Assumes x and z dimensions match between zone and full mesh
        // (zone trims only in y for now).
        int fullNx = fftDimensions_.x();
        int fullNz = fftDimensions_.z();
        label nTotalCells = matrix_.diag().size();
        int fullNy = nTotalCells / (fullNx * fullNz);
        int fullNxy = fullNx * fullNy;

        // Build zone membership lookup for face filtering
        boolList inZone(nTotalCells, false);
        forAll(zoneCellsSorted_, i)
        {
            inZone[zoneCellsSorted_[i]] = true;
        }

        double sumCoeffX = 0, sumCoeffY = 0, sumCoeffZ = 0;
        int countX = 0, countY = 0, countZ = 0;

        forAll(owner, facei)
        {
            label o = owner[facei];
            label n = neighbour[facei];

            // Only count zone-internal faces (both cells in zone)
            if (!inZone[o] || !inZone[n]) continue;

            int diff = n - o;  // always positive (neighbour > owner)
            double val = Foam::mag(upper[facei]);

            if (diff == 1)
            {
                sumCoeffX += val;
                countX++;
            }
            else if (diff == fullNx)
            {
                sumCoeffY += val;
                countY++;
            }
            else if (diff == fullNxy)
            {
                sumCoeffZ += val;
                countZ++;
            }
        }

        double meanCoeffX = (countX > 0) ? sumCoeffX / countX : 0;
        double meanCoeffY = (countY > 0) ? sumCoeffY / countY : 0;
        double meanCoeffZ = (countZ > 0) ? sumCoeffZ / countZ : 0;

        if (debug_ >= 1)
        {
            Info<< "OGLSpectral: zone ldu couplings:"
                << " coeffX=" << meanCoeffX
                << " (n=" << countX << ")"
                << " coeffY=" << meanCoeffY
                << " (n=" << countY << ")"
                << " coeffZ=" << meanCoeffZ
                << " (n=" << countZ << ")"
                << " (fullMesh: " << fullNx
                << "x" << fullNy << "x" << fullNz << ")"
                << endl;
        }

        fftSolver->updateCoeffs(meanCoeffX, meanCoeffY, meanCoeffZ);
        return;
    }

    // Extract mean coupling coefficients from the CSR matrix and update
    // the FFT eigenvalues to match the actual assembled pressure matrix.
    // This accounts for rAU (1/Ap) scaling from the momentum equation.
    auto csrMtx = gko::as<gko::matrix::Csr<ValueType, int>>(
        op->localMatrix()
    );
    auto nRows = csrMtx->get_size()[0];
    auto rowPtrs = csrMtx->get_const_row_ptrs();
    auto colIdxs = csrMtx->get_const_col_idxs();
    auto vals = csrMtx->get_const_values();

    std::vector<int> hRowPtrs(nRows + 1);
    std::vector<int> hColIdxs(csrMtx->get_num_stored_elements());
    std::vector<ValueType> hVals(csrMtx->get_num_stored_elements());

    exec->get_master()->copy_from(
        exec.get(), nRows + 1, rowPtrs, hRowPtrs.data()
    );
    exec->get_master()->copy_from(
        exec.get(),
        csrMtx->get_num_stored_elements(),
        colIdxs, hColIdxs.data()
    );
    exec->get_master()->copy_from(
        exec.get(),
        csrMtx->get_num_stored_elements(),
        vals, hVals.data()
    );

    // For structured mesh: x-neighbors differ by +/-1,
    // y-neighbors by +/-nx, z-neighbors by +/-(nx*ny)
    int nx = fftDimensions_.x();
    int nxy = fftDimensions_.x() * fftDimensions_.y();

    double sumCoeffX = 0, sumCoeffY = 0, sumCoeffZ = 0;
    int countX = 0, countY = 0, countZ = 0;

    for (gko::size_type row = 0; row < nRows; row++)
    {
        for (int j = hRowPtrs[row]; j < hRowPtrs[row+1]; j++)
        {
            int col = hColIdxs[j];
            if (col == static_cast<int>(row)) continue;

            int diff = std::abs(col - static_cast<int>(row));
            double val = std::abs(static_cast<double>(hVals[j]));

            if (diff == 1)
            {
                sumCoeffX += val;
                countX++;
            }
            else if (diff == nx)
            {
                sumCoeffY += val;
                countY++;
            }
            else if (diff == nxy)
            {
                sumCoeffZ += val;
                countZ++;
            }
        }
    }

    double meanCoeffX = (countX > 0) ? sumCoeffX / countX : 0;
    double meanCoeffY = (countY > 0) ? sumCoeffY / countY : 0;
    double meanCoeffZ = (countZ > 0) ? sumCoeffZ / countZ : 0;

    if (debug_ >= 1)
    {
        Info<< "OGLSpectral: mean CSR couplings:"
            << " coeffX=" << meanCoeffX
            << " (n=" << countX << ")"
            << " coeffY=" << meanCoeffY
            << " (n=" << countY << ")"
            << " coeffZ=" << meanCoeffZ
            << " (n=" << countZ << ")"
            << endl;
    }

    fftSolver->updateCoeffs(meanCoeffX, meanCoeffY, meanCoeffZ);
}


template<typename ValueType>
Foam::label Foam::OGL::OGLSpectralSolver::spectralSolveImpl
(
    scalarField& psi,
    const scalarField& source,
    std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op,
    std::shared_ptr<FFTPreconditioner<ValueType>>& fftSolver
) const
{
    auto exec = OGLExecutor::instance().executor();

    try
    {
        // Ensure operator has current matrix values
        ensureOperator(op);

        // Initialize or update FFT eigenvalues from the CSR matrix.
        // On the first call, this creates the cuFFT plans and eigenvalue
        // arrays. On subsequent calls, it updates eigenvalues if rAU changed.
        if (!coeffsInitialized_)
        {
            initFFTSolver(op, fftSolver);
            coeffsInitialized_ = true;
        }
        else
        {
            // Update CSR values for the operator (matrix may have changed)
            // and re-extract coupling coefficients
            initFFTSolver(op, fftSolver);
        }

        // Convert source to Ginkgo Dense vector on GPU
        std::shared_ptr<gko::matrix::Dense<ValueType>> b;
        std::shared_ptr<gko::matrix::Dense<ValueType>> x;

        if constexpr (std::is_same<ValueType, float>::value)
        {
            b = FP32CastWrapper::toGinkgoF32(exec, source);
            x = FP32CastWrapper::toGinkgoF32(exec, psi);
        }
        else
        {
            b = FP32CastWrapper::toGinkgoF64(exec, source);
            x = FP32CastWrapper::toGinkgoF64(exec, psi);
        }

        // Apply DCT direct solve: x = L^{-1} * b
        // This is O(N log N) via cuFFT — a single pass, no iteration.
        exec->synchronize();
        auto solveStart = std::chrono::high_resolution_clock::now();

        fftSolver->apply(b.get(), x.get());

        OGLExecutor::instance().synchronize();
        auto solveEnd = std::chrono::high_resolution_clock::now();

        double solveMs = std::chrono::duration<double, std::milli>(
            solveEnd - solveStart
        ).count();

        if (debug_ >= 1)
        {
            Info<< "OGLSpectral: DCT direct solve: " << solveMs << " ms"
                << endl;
        }

        // Copy solution back to host
        if constexpr (std::is_same<ValueType, float>::value)
        {
            FP32CastWrapper::fromGinkgoF32(x.get(), psi);
        }
        else
        {
            FP32CastWrapper::fromGinkgoF64(x.get(), psi);
        }

        // Numerical safety: check for NaN/Inf
        forAll(psi, i)
        {
            if (std::isnan(psi[i]))
            {
                FatalErrorInFunction
                    << "OGLSpectral: NaN detected in solution"
                    << abort(FatalError);
            }
        }
    }
    catch (const std::exception& e)
    {
        OGLExecutor::checkGinkgoError("spectralSolveImpl", e);
    }

    // Direct solve: always 1 "iteration"
    return 1;
}


template<typename ValueType>
void Foam::OGL::OGLSpectralSolver::applyDCT
(
    scalarField& psi,
    const scalarField& source,
    std::shared_ptr<FFTPreconditioner<ValueType>>& fftSolver
) const
{
    auto exec = OGLExecutor::instance().executor();

    // Convert source (RHS) and psi (solution) to Ginkgo Dense on GPU
    std::shared_ptr<gko::matrix::Dense<ValueType>> b;
    std::shared_ptr<gko::matrix::Dense<ValueType>> x;

    if constexpr (std::is_same<ValueType, float>::value)
    {
        b = FP32CastWrapper::toGinkgoF32(exec, source);
        x = FP32CastWrapper::toGinkgoF32(exec, psi);
    }
    else
    {
        b = FP32CastWrapper::toGinkgoF64(exec, source);
        x = FP32CastWrapper::toGinkgoF64(exec, psi);
    }

    // Apply DCT direct solve: x = L^{-1} * b
    fftSolver->apply(b.get(), x.get());

    // Copy solution back to host
    if constexpr (std::is_same<ValueType, float>::value)
    {
        FP32CastWrapper::fromGinkgoF32(x.get(), psi);
    }
    else
    {
        FP32CastWrapper::fromGinkgoF64(x.get(), psi);
    }
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

Foam::label Foam::OGL::OGLSpectralSolver::solveFP32
(
    scalarField& psi,
    const scalarField& source,
    const scalar tolerance
) const
{
    label iters = spectralSolveImpl<float>(
        psi, source, operatorF32_, fftSolverF32_
    );

    // Persist to static cache
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[fieldName_].opF32 = operatorF32_;
        cache_[fieldName_].fftF32 = fftSolverF32_;
        cache_[fieldName_].coeffsInitialized = coeffsInitialized_;
    }

    return iters;
}


Foam::label Foam::OGL::OGLSpectralSolver::solveFP64
(
    scalarField& psi,
    const scalarField& source,
    const scalar tolerance
) const
{
    label iters = spectralSolveImpl<double>(
        psi, source, operatorF64_, fftSolverF64_
    );

    // Persist to static cache
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[fieldName_].opF64 = operatorF64_;
        cache_[fieldName_].fftF64 = fftSolverF64_;
        cache_[fieldName_].coeffsInitialized = coeffsInitialized_;
    }

    return iters;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::OGL::OGLSpectralSolver::OGLSpectralSolver
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
    operatorF32_(nullptr),
    operatorF64_(nullptr),
    fftSolverF32_(nullptr),
    fftSolverF64_(nullptr),
    coeffsInitialized_(false),
    spectralZoneName_(),
    useZone_(false),
    zoneInitialized_(false),
    overlapWidth_(0),
    extFftDims_(0, 0, 0),
    extFftF32_(nullptr),
    extFftF64_(nullptr)
{
    // Default to more refinement iterations than base class (3).
    // The spectral solver needs ~5-7 iterations to drive boundary rAU
    // mismatch below tolerance (rho^k convergence, rho ~ 0.15).
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

    // OGLSolverBase only reads fftDimensions/meshSpacing when preconditioner
    // is FFT or FFT_BLOCK_JACOBI. The spectral solver always needs them,
    // regardless of preconditioner setting, so read them here.
    if (controlDict_.found("OGLCoeffs"))
    {
        const dictionary& oglDict = controlDict_.subDict("OGLCoeffs");

        if (oglDict.found("fftDimensions"))
        {
            fftDimensions_ = oglDict.lookup<Vector<label>>("fftDimensions");
        }
        if (oglDict.found("meshSpacing"))
        {
            meshSpacing_ = oglDict.lookup<Vector<scalar>>("meshSpacing");
        }
        if (oglDict.found("spectralZone"))
        {
            spectralZoneName_ = oglDict.lookup<word>("spectralZone");
            useZone_ = true;
        }
        if (oglDict.found("overlapWidth"))
        {
            overlapWidth_ = oglDict.lookup<label>("overlapWidth");
        }
    }

    // Auto-detect or validate FFT dimensions
    bool autoDetected = false;
    if
    (
        fftDimensions_.x() <= 0
     || fftDimensions_.y() <= 0
     || fftDimensions_.z() <= 0
    )
    {
        if (useZone_)
        {
            FatalErrorInFunction
                << "OGLSpectral: fftDimensions required when using "
                << "spectralZone (auto-detect not supported for zones)"
                << abort(FatalError);
        }

        if (!detectStructuredMesh())
        {
            FatalErrorInFunction
                << "OGLSpectral: fftDimensions not specified and "
                << "auto-detection failed. Specify fftDimensions in "
                << "OGLCoeffs."
                << abort(FatalError);
        }
        autoDetected = true;
    }

    // meshSpacing: set dummy values if not specified.
    // updateCoeffs will override with actual coupling coefficients.
    if
    (
        meshSpacing_.x() <= 0
     || meshSpacing_.y() <= 0
     || meshSpacing_.z() <= 0
    )
    {
        meshSpacing_ = Vector<scalar>(1.0, 1.0, 1.0);
    }

    // Restore cached state from previous instantiation
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto it = cache_.find(fieldName);
        if (it != cache_.end())
        {
            operatorF32_ = it->second.opF32;
            operatorF64_ = it->second.opF64;
            fftSolverF32_ = it->second.fftF32;
            fftSolverF64_ = it->second.fftF64;
            extFftF32_ = it->second.extFftF32;
            extFftF64_ = it->second.extFftF64;
            coeffsInitialized_ = it->second.coeffsInitialized;

            if (debug_ >= 2)
            {
                Info<< "OGLSpectral: Restored cached FFT solver for "
                    << fieldName << endl;
            }
        }
    }

    Info<< "OGLSpectral: Created for field " << fieldName << nl
        << "  grid: "
        << fftDimensions_.x() << " x "
        << fftDimensions_.y() << " x "
        << fftDimensions_.z()
        << " = " << (fftDimensions_.x() * fftDimensions_.y()
                      * fftDimensions_.z())
        << " cells"
        << (autoDetected ? " (auto-detected)" : "") << nl
        << "  spacing: ("
        << meshSpacing_.x() << ", "
        << meshSpacing_.y() << ", "
        << meshSpacing_.z() << ")" << nl
        << "  precisionPolicy: "
        << (precisionPolicy_ == PrecisionPolicy::FP64 ? "FP64" :
            precisionPolicy_ == PrecisionPolicy::FP32 ? "FP32" : "MIXED")
        << nl
        << "  iterativeRefinement: " << iterativeRefinement_
        << nl
        << "  spectralZone: "
        << (useZone_ ? spectralZoneName_ : word("none (full mesh)"))
        << nl
        << "  overlapWidth: " << overlapWidth_
        << (overlapWidth_ > 0 ? " (Restricted Additive Schwarz)" : "")
        << endl;
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

bool Foam::OGL::OGLSpectralSolver::detectStructuredMesh()
{
    const auto& addr = matrix_.lduAddr();
    const labelUList& owner = addr.lowerAddr();
    const labelUList& neighbour = addr.upperAddr();

    if (!matrix_.hasUpper())
    {
        return false;
    }

    const scalarField& upper = matrix_.upper();
    label nCells = matrix_.diag().size();

    // Build histogram: stride -> (sumCoeff, count)
    std::map<label, std::pair<double, label>> strideHist;

    forAll(owner, facei)
    {
        label diff = neighbour[facei] - owner[facei];
        double val = Foam::mag(upper[facei]);
        auto& entry = strideHist[diff];
        entry.first += val;
        entry.second++;
    }

    // Sort strides by frequency (descending)
    std::vector<std::pair<label, label>> stridesByFreq;
    for (const auto& kv : strideHist)
    {
        stridesByFreq.push_back({kv.second.second, kv.first});
    }
    std::sort(stridesByFreq.rbegin(), stridesByFreq.rend());

    if (stridesByFreq.size() < 2)
    {
        return false;
    }

    // Verify top strides account for nearly all faces (structured mesh)
    label totalFaces = owner.size();
    label topFaces = 0;
    size_t nDirs = std::min(stridesByFreq.size(), size_t(3));
    for (size_t i = 0; i < nDirs; i++)
    {
        topFaces += stridesByFreq[i].first;
    }

    if (topFaces < label(0.95 * totalFaces))
    {
        WarningInFunction
            << "Top " << nDirs << " strides account for only "
            << (100.0 * topFaces / totalFaces)
            << "% of faces. Mesh is not structured." << endl;
        return false;
    }

    // Extract strides sorted by value
    std::vector<label> topStrides;
    for (size_t i = 0; i < nDirs; i++)
    {
        topStrides.push_back(stridesByFreq[i].second);
    }
    std::sort(topStrides.begin(), topStrides.end());

    // Smallest stride must be 1 (x-direction in blockMesh ordering)
    if (topStrides[0] != 1)
    {
        WarningInFunction
            << "Smallest stride is " << topStrides[0]
            << " (expected 1). Not a standard structured mesh." << endl;
        return false;
    }

    // Infer dimensions: stride_y = nx, stride_z = nx*ny
    label nx = topStrides[1];
    label ny, nz;

    if (topStrides.size() >= 3)
    {
        label strideZ = topStrides[2];
        ny = strideZ / nx;
        nz = nCells / (nx * ny);
    }
    else
    {
        // 2D case
        ny = nCells / nx;
        nz = 1;
    }

    if (nx * ny * nz != nCells)
    {
        WarningInFunction
            << "Detected " << nx << "x" << ny << "x" << nz
            << " = " << (nx*ny*nz)
            << " != " << nCells << " cells" << endl;
        return false;
    }

    // Validate face counts match expected structured mesh
    label expX = (nx - 1) * ny * nz;
    label expY = nx * (ny - 1) * nz;
    label expZ = (nz > 1) ? nx * ny * (nz - 1) : 0;

    label actX = strideHist.count(1) ? strideHist[1].second : 0;
    label actY = strideHist.count(nx) ? strideHist[nx].second : 0;
    label actZ = (nz > 1 && strideHist.count(nx*ny))
                 ? strideHist[nx*ny].second : 0;

    if (actX != expX || actY != expY || (nz > 1 && actZ != expZ))
    {
        WarningInFunction
            << "Face count mismatch for " << nx << "x" << ny << "x" << nz
            << ". Expected (" << expX << "," << expY << "," << expZ << ")"
            << " got (" << actX << "," << actY << "," << actZ << ")"
            << endl;
        return false;
    }

    fftDimensions_ = Vector<label>(nx, ny, nz);

    // Compute mesh spacing from coupling coefficients.
    // FV Laplacian: coeffX = dy*dz/dx, coeffY = dx*dz/dy, coeffZ = dx*dy/dz
    // => dx = sqrt(coeffY*coeffZ), dy = sqrt(coeffX*coeffZ), dz = sqrt(coeffX*coeffY)
    double meanCoeffX = strideHist[1].first / strideHist[1].second;
    double meanCoeffY = strideHist[nx].first / strideHist[nx].second;

    if (nz > 1 && strideHist.count(nx*ny) && strideHist[nx*ny].second > 0)
    {
        double meanCoeffZ =
            strideHist[nx*ny].first / strideHist[nx*ny].second;

        meshSpacing_.x() = std::sqrt(meanCoeffY * meanCoeffZ);
        meshSpacing_.y() = std::sqrt(meanCoeffX * meanCoeffZ);
        meshSpacing_.z() = std::sqrt(meanCoeffX * meanCoeffY);
    }
    else
    {
        // 2D: spacing for initial eigenvalues (overridden by updateCoeffs)
        meshSpacing_ = Vector<scalar>(1.0, 1.0, 1.0);
    }

    Info<< "OGLSpectral: Auto-detected structured mesh: "
        << nx << " x " << ny << " x " << nz
        << " = " << nCells << " cells" << nl
        << "  spacing: ("
        << meshSpacing_.x() << ", "
        << meshSpacing_.y() << ", "
        << meshSpacing_.z() << ")" << nl
        << "  mean couplings: x=" << meanCoeffX
        << " y=" << meanCoeffY;

    if (nz > 1 && strideHist.count(nx*ny))
    {
        Info<< " z="
            << strideHist[nx*ny].first / strideHist[nx*ny].second;
    }

    Info<< endl;

    return true;
}


void Foam::OGL::OGLSpectralSolver::initZone() const
{
    // The lduMesh interface gives us access to cellZones via the fvMesh
    const auto& meshRef = matrix().mesh();

    // Find zone by name via the objectRegistry/polyMesh interface
    const auto& cellZones =
        dynamic_cast<const polyMesh&>(meshRef).cellZones();

    label zoneID = cellZones.findIndex(spectralZoneName_);
    if (zoneID < 0)
    {
        FatalErrorInFunction
            << "cellZone '" << spectralZoneName_ << "' not found"
            << abort(FatalError);
    }

    const labelList& zoneCells = cellZones[zoneID];

    // Sort for correct (i,j,k) DCT ordering — blockMesh numbers cells
    // as i + nx*j + nx*ny*k, so sorted global indices give the local
    // (i',j',k') ordering that the DCT expects.
    zoneCellsSorted_ = zoneCells;
    sort(zoneCellsSorted_);

    // Build non-zone cell list
    // lduMatrix size gives the number of cells (rows)
    label nTotalCells = matrix_.diag().size();
    boolList inZone(nTotalCells, false);
    forAll(zoneCellsSorted_, i)
    {
        inZone[zoneCellsSorted_[i]] = true;
    }

    nonZoneCells_.setSize(nTotalCells - zoneCellsSorted_.size());
    label j = 0;
    forAll(inZone, celli)
    {
        if (!inZone[celli])
        {
            nonZoneCells_[j++] = celli;
        }
    }

    // Validate zone size matches FFT dimensions
    label expectedZoneCells =
        fftDimensions_.x() * fftDimensions_.y() * fftDimensions_.z();

    if (zoneCellsSorted_.size() != expectedZoneCells)
    {
        FatalErrorInFunction
            << "cellZone '" << spectralZoneName_ << "' has "
            << zoneCellsSorted_.size() << " cells, but fftDimensions "
            << fftDimensions_ << " requires "
            << expectedZoneCells << " cells"
            << abort(FatalError);
    }

    zoneInitialized_ = true;

    Info<< "OGLSpectral: Zone '" << spectralZoneName_ << "': "
        << zoneCellsSorted_.size() << " spectral cells, "
        << nonZoneCells_.size() << " Jacobi cells" << endl;

    // --- Overlap extension for Restricted Additive Schwarz ---
    if (overlapWidth_ > 0)
    {
        // Detect full-mesh structured strides from ldu face addressing
        const auto& addr = matrix_.lduAddr();
        const labelUList& owner = addr.lowerAddr();
        const labelUList& neighbour = addr.upperAddr();

        std::map<label, label> strideCount;
        forAll(owner, facei)
        {
            strideCount[neighbour[facei] - owner[facei]]++;
        }

        // Sort strides by value, take top 2-3
        std::vector<label> strides;
        for (const auto& kv : strideCount)
        {
            strides.push_back(kv.first);
        }
        std::sort(strides.begin(), strides.end());

        label fullNx = strides[1];
        label fullNy = (strides.size() >= 3)
            ? strides[2] / fullNx
            : nTotalCells / fullNx;
        label fullNz = nTotalCells / (fullNx * fullNy);

        // Find bounding structured index range of zone cells
        label ixMin = fullNx, iyMin = fullNy, izMin = fullNz;
        label ixMax = 0, iyMax = 0, izMax = 0;

        forAll(zoneCellsSorted_, ci)
        {
            label c = zoneCellsSorted_[ci];
            label ix = c % fullNx;
            label iy = (c / fullNx) % fullNy;
            label iz = c / (fullNx * fullNy);

            ixMin = min(ixMin, ix); ixMax = max(ixMax, ix);
            iyMin = min(iyMin, iy); iyMax = max(iyMax, iy);
            izMin = min(izMin, iz); izMax = max(izMax, iz);
        }

        // Extend by overlapWidth, clamped to full mesh bounds
        label exMin = max(label(0), ixMin - overlapWidth_);
        label exMax = min(fullNx - 1, ixMax + overlapWidth_);
        label eyMin = max(label(0), iyMin - overlapWidth_);
        label eyMax = min(fullNy - 1, iyMax + overlapWidth_);
        label ezMin = max(label(0), izMin - overlapWidth_);
        label ezMax = min(fullNz - 1, izMax + overlapWidth_);

        label extNx = exMax - exMin + 1;
        label extNy = eyMax - eyMin + 1;
        label extNz = ezMax - ezMin + 1;

        extFftDims_ = Vector<label>(extNx, extNy, extNz);

        // Build extended zone cell list and zone-interior mask
        label extNcells = extNx * extNy * extNz;
        extendedZoneCells_.setSize(extNcells);
        extToZoneMap_.setSize(extNcells, -1);

        label idx = 0;
        for (label iz = ezMin; iz <= ezMax; iz++)
        {
            for (label iy = eyMin; iy <= eyMax; iy++)
            {
                for (label ix = exMin; ix <= exMax; ix++)
                {
                    label globalCell =
                        ix + fullNx * iy + fullNx * fullNy * iz;
                    extendedZoneCells_[idx] = globalCell;

                    // Mark zone-interior cells (scatter these back)
                    if (inZone[globalCell])
                    {
                        extToZoneMap_[idx] = 1;
                    }
                    idx++;
                }
            }
        }

        Info<< "OGLSpectral: RAS overlap=" << overlapWidth_
            << ": extended zone "
            << extNx << "x" << extNy << "x" << extNz
            << " = " << extNcells << " cells"
            << " (zone=" << zoneCellsSorted_.size()
            << " overlap=" << (extNcells - zoneCellsSorted_.size())
            << ")" << endl;
    }
}


void Foam::OGL::OGLSpectralSolver::applyZonePreconditioner
(
    scalarField& z,
    const scalarField& r
) const
{
    const label nZone = zoneCellsSorted_.size();

    // --- Zone cells: gather → DCT → scatter ---
    scalarField zoneR(nZone);
    forAll(zoneCellsSorted_, i)
    {
        zoneR[i] = r[zoneCellsSorted_[i]];
    }

    scalarField zoneZ(nZone, 0.0);
    try
    {
        if (precisionPolicy_ == PrecisionPolicy::FP64)
        {
            applyDCT<double>(zoneZ, zoneR, fftSolverF64_);
        }
        else
        {
            applyDCT<float>(zoneZ, zoneR, fftSolverF32_);
        }
    }
    catch (const std::exception& e)
    {
        OGLExecutor::checkGinkgoError("spectral zone DCT", e);
    }

    forAll(zoneCellsSorted_, i)
    {
        z[zoneCellsSorted_[i]] = zoneZ[i];
    }

    // --- Non-zone cells: diagonal (Jacobi) preconditioning ---
    const scalarField& diag = matrix_.diag();
    forAll(nonZoneCells_, i)
    {
        label c = nonZoneCells_[i];
        z[c] = r[c] / diag[c];
    }
}


void Foam::OGL::OGLSpectralSolver::applyOverlapAS
(
    scalarField& z,
    const scalarField& r
) const
{
    const scalarField& diag = matrix_.diag();

    // Start with Jacobi preconditioning everywhere.
    // Zone-interior cells will be overwritten with DCT result.
    forAll(z, i)
    {
        z[i] = r[i] / diag[i];
    }

    // Gather residual from extended zone cells
    const label nExt = extendedZoneCells_.size();
    scalarField extR(nExt);
    forAll(extendedZoneCells_, i)
    {
        extR[i] = r[extendedZoneCells_[i]];
    }

    // DCT solve on extended zone
    scalarField extZ(nExt, 0.0);
    try
    {
        if (precisionPolicy_ == PrecisionPolicy::FP64)
        {
            applyDCT<double>(extZ, extR, extFftF64_);
        }
        else
        {
            applyDCT<float>(extZ, extR, extFftF32_);
        }
    }
    catch (const std::exception& e)
    {
        OGLExecutor::checkGinkgoError("spectral RAS DCT", e);
    }

    // Additive Schwarz scatter: write ALL extended zone cells.
    // This makes M^{-1} = P_ext * DCT^{-1} * P_ext + P_comp * D^{-1} * P_comp
    // which is SPD (required for CG). Non-extended cells keep Jacobi values.
    forAll(extendedZoneCells_, i)
    {
        z[extendedZoneCells_[i]] = extZ[i];
    }
}


Foam::solverPerformance Foam::OGL::OGLSpectralSolver::solve
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

    // --- Compute A*psi once for normFactor + initial residual (1 SpMV)
    scalarField wA(nCells);
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // Norm factor (same computation as OpenFOAM's PCG.C)
    scalarField tmpField(nCells);
    scalar normFactor = this->normFactor(psi, source, wA, tmpField);

    // Initial residual: r = source - A*psi (reuse wA from above)
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

    // --- Zone initialization (once)
    if (useZone_ && !zoneInitialized_)
    {
        initZone();
    }

    // --- Setup phase: create FFT solver and calibrate eigenvalues.
    //     In full-mesh mode, extracts mean CSR couplings from the pressure
    //     matrix. In zone mode, uses analytical eigenvalues from spacing.
    //     Only done ONCE per field — cached for subsequent solves.
    if (!coeffsInitialized_)
    {
        auto setupStart = std::chrono::high_resolution_clock::now();

        try
        {
            if (useZone_ && overlapWidth_ > 0)
            {
                // RAS mode: create extended FFT solver with extended dims
                auto exec = OGLExecutor::instance().executor();
                auto extN = static_cast<gko::size_type>(
                    extFftDims_.x() * extFftDims_.y() * extFftDims_.z()
                );

                if (precisionPolicy_ == PrecisionPolicy::FP64)
                {
                    extFftF64_ = gko::share(
                        FFTPreconditioner<double>::create(
                            exec,
                            gko::dim<2>{extN, extN},
                            extFftDims_.x(),
                            extFftDims_.y(),
                            extFftDims_.z(),
                            double(meshSpacing_.x()),
                            double(meshSpacing_.y()),
                            double(meshSpacing_.z())
                        )
                    );
                }
                else
                {
                    extFftF32_ = gko::share(
                        FFTPreconditioner<float>::create(
                            exec,
                            gko::dim<2>{extN, extN},
                            extFftDims_.x(),
                            extFftDims_.y(),
                            extFftDims_.z(),
                            double(meshSpacing_.x()),
                            double(meshSpacing_.y()),
                            double(meshSpacing_.z())
                        )
                    );
                }

                // Extract coupling coefficients from faces in extended zone
                const auto& addr = matrix_.lduAddr();
                const labelUList& owner = addr.lowerAddr();
                const labelUList& neighbour = addr.upperAddr();
                const scalarField& upper = matrix_.upper();

                label nTotal = matrix_.diag().size();

                // Detect full-mesh strides
                std::map<label, label> strideCount;
                forAll(owner, facei)
                {
                    strideCount[neighbour[facei] - owner[facei]]++;
                }
                std::vector<label> strides;
                for (const auto& kv : strideCount)
                {
                    strides.push_back(kv.first);
                }
                std::sort(strides.begin(), strides.end());
                label fullNx = strides[1];
                label fullNy = (strides.size() >= 3)
                    ? strides[2] / fullNx
                    : nTotal / fullNx;
                int fullNxy = fullNx * fullNy;

                // Build extended zone membership
                boolList inExt(nTotal, false);
                forAll(extendedZoneCells_, i)
                {
                    inExt[extendedZoneCells_[i]] = true;
                }

                double sumCX = 0, sumCY = 0, sumCZ = 0;
                int cntX = 0, cntY = 0, cntZ = 0;

                forAll(owner, facei)
                {
                    label o = owner[facei];
                    label n = neighbour[facei];
                    if (!inExt[o] || !inExt[n]) continue;

                    int diff = n - o;
                    double val = Foam::mag(upper[facei]);

                    if (diff == 1) { sumCX += val; cntX++; }
                    else if (diff == fullNx) { sumCY += val; cntY++; }
                    else if (diff == fullNxy) { sumCZ += val; cntZ++; }
                }

                double mCX = (cntX > 0) ? sumCX / cntX : 0;
                double mCY = (cntY > 0) ? sumCY / cntY : 0;
                double mCZ = (cntZ > 0) ? sumCZ / cntZ : 0;

                if (debug_ >= 1)
                {
                    Info<< "OGLSpectral: RAS extended zone couplings:"
                        << " coeffX=" << mCX << " (n=" << cntX << ")"
                        << " coeffY=" << mCY << " (n=" << cntY << ")"
                        << " coeffZ=" << mCZ << " (n=" << cntZ << ")"
                        << endl;
                }

                if (precisionPolicy_ == PrecisionPolicy::FP64)
                {
                    extFftF64_->updateCoeffs(mCX, mCY, mCZ);
                }
                else
                {
                    extFftF32_->updateCoeffs(mCX, mCY, mCZ);
                }
            }
            else if (useZone_)
            {
                // Non-overlapping zone: create FFT solver with zone dims,
                // extract coefficients from ldu faces
                if (precisionPolicy_ == PrecisionPolicy::FP64)
                {
                    initFFTSolver(operatorF64_, fftSolverF64_);
                }
                else
                {
                    initFFTSolver(operatorF32_, fftSolverF32_);
                }
            }
            else
            {
                // Full-mesh mode: extract CSR coupling coefficients
                if (precisionPolicy_ == PrecisionPolicy::FP64)
                {
                    ensureOperator(operatorF64_);
                    initFFTSolver(operatorF64_, fftSolverF64_);
                }
                else
                {
                    ensureOperator(operatorF32_);
                    initFFTSolver(operatorF32_, fftSolverF32_);
                }
            }
            coeffsInitialized_ = true;
        }
        catch (const std::exception& e)
        {
            OGLExecutor::checkGinkgoError("spectral setup", e);
        }

        auto setupEnd = std::chrono::high_resolution_clock::now();
        double setupMs = std::chrono::duration<double, std::milli>(
            setupEnd - setupStart
        ).count();

        Info<< "OGLSpectral: coefficient setup: " << setupMs << " ms"
            << " (cached for subsequent solves)" << endl;
    }

    // --- Preconditioned CG with DCT preconditioner.
    //
    //     Per iteration: 1 SpMV + 1 DCT + 2 dot products + 3 AXPY
    //     Convergence check uses recurrence residual (no extra SpMV),
    //     matching OpenFOAM's PCG.C pattern.

    // Pre-allocate working vectors (outside loop)
    scalarField zField(nCells, 0.0);
    scalarField pField(nCells, 0.0);
    scalarField qField(nCells);

    // z = M^{-1} * r (initial preconditioner application)
    if (useZone_ && overlapWidth_ > 0)
    {
        applyOverlapAS(zField, rField);
    }
    else if (useZone_)
    {
        applyZonePreconditioner(zField, rField);
    }
    else
    {
        try
        {
            if (precisionPolicy_ == PrecisionPolicy::FP64)
            {
                applyDCT<double>(zField, rField, fftSolverF64_);
            }
            else
            {
                applyDCT<float>(zField, rField, fftSolverF32_);
            }
        }
        catch (const std::exception& e)
        {
            OGLExecutor::checkGinkgoError("spectral PCG init", e);
        }
    }

    // p = z
    forAll(pField, i) { pField[i] = zField[i]; }

    // rz = (r, z)
    scalar rz = gSumProd(rField, zField, matrix().mesh().comm());

    label totalIters = 0;

    do
    {
        // q = A * p
        matrix_.Amul(qField, pField, interfaceBouCoeffs_, interfaces_, cmpt);

        // alpha = (r, z) / (p, q)
        scalar pq = gSumProd(pField, qField, matrix().mesh().comm());

        // Singularity check (same as PCG.C)
        if (solverPerf.checkSingularity(mag(pq) / normFactor))
        {
            break;
        }

        scalar alpha = rz / pq;

        // x += alpha * p,  r -= alpha * q
        forAll(psi, i)
        {
            psi[i] += alpha * pField[i];
            rField[i] -= alpha * qField[i];
        }

        totalIters++;

        // Convergence from recurrence residual (no extra SpMV!)
        solverPerf.finalResidual() =
            gSumMag(rField, matrix().mesh().comm()) / normFactor;

        if (debug_ >= 1)
        {
            Info<< "OGLSpectral: PCG Iter " << totalIters
                << ", residual = " << solverPerf.finalResidual() << endl;
        }

        // z = M^{-1} * r
        forAll(zField, i) { zField[i] = 0.0; }
        if (useZone_ && overlapWidth_ > 0)
        {
            applyOverlapAS(zField, rField);
        }
        else if (useZone_)
        {
            applyZonePreconditioner(zField, rField);
        }
        else
        {
            try
            {
                if (precisionPolicy_ == PrecisionPolicy::FP64)
                {
                    applyDCT<double>(zField, rField, fftSolverF64_);
                }
                else
                {
                    applyDCT<float>(zField, rField, fftSolverF32_);
                }
            }
            catch (const std::exception& e)
            {
                OGLExecutor::checkGinkgoError("spectral PCG precond", e);
            }
        }

        // beta = (r_new, z_new) / (r_old, z_old)
        scalar rzNew = gSumProd(rField, zField, matrix().mesh().comm());
        scalar beta = rzNew / (rz + VSMALL);
        rz = rzNew;

        // p = z + beta * p
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

    // Persist cached GPU state for reuse across solver instantiations
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto& entry = cache_[fieldName_];
        entry.opF32 = operatorF32_;
        entry.opF64 = operatorF64_;
        entry.fftF32 = fftSolverF32_;
        entry.fftF64 = fftSolverF64_;
        entry.extFftF32 = extFftF32_;
        entry.extFftF64 = extFftF64_;
        entry.coeffsInitialized = coeffsInitialized_;
    }

    return solverPerf;
}


// ************************************************************************* //
