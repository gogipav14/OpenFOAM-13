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

#include <chrono>
#include <cmath>
#include <limits>

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
    coeffsInitialized_(false)
{
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
    }

    // Validate that FFT dimensions are provided
    if
    (
        fftDimensions_.x() <= 0
     || fftDimensions_.y() <= 0
     || fftDimensions_.z() <= 0
    )
    {
        FatalErrorInFunction
            << "OGLSpectral requires fftDimensions in OGLCoeffs. "
            << "Got: (" << fftDimensions_.x()
            << " " << fftDimensions_.y()
            << " " << fftDimensions_.z() << ")"
            << abort(FatalError);
    }

    if
    (
        meshSpacing_.x() <= 0
     || meshSpacing_.y() <= 0
     || meshSpacing_.z() <= 0
    )
    {
        FatalErrorInFunction
            << "OGLSpectral requires meshSpacing in OGLCoeffs. "
            << "Got: (" << meshSpacing_.x()
            << " " << meshSpacing_.y()
            << " " << meshSpacing_.z() << ")"
            << abort(FatalError);
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
        << " cells" << nl
        << "  spacing: ("
        << meshSpacing_.x() << ", "
        << meshSpacing_.y() << ", "
        << meshSpacing_.z() << ")" << nl
        << "  precisionPolicy: "
        << (precisionPolicy_ == PrecisionPolicy::FP64 ? "FP64" :
            precisionPolicy_ == PrecisionPolicy::FP32 ? "FP32" : "MIXED")
        << nl
        << "  iterativeRefinement: " << iterativeRefinement_
        << endl;
}


// ************************************************************************* //
