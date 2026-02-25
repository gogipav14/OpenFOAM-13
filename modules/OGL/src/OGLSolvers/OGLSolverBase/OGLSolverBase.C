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

#include "OGLSolverBase.H"
#include "OGLExecutor.H"

// * * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * //

void Foam::OGL::OGLSolverBase::readOGLControls()
{
    // Read OGL-specific controls from subdictionary
    if (controlDict_.found("OGLCoeffs"))
    {
        const dictionary& oglDict = controlDict_.subDict("OGLCoeffs");

        // Precision policy
        word precisionStr = oglDict.lookupOrDefault<word>
        (
            "precisionPolicy",
            "FP32"
        );

        if (precisionStr == "FP64")
        {
            precisionPolicy_ = PrecisionPolicy::FP64;
        }
        else if (precisionStr == "FP32")
        {
            precisionPolicy_ = PrecisionPolicy::FP32;
        }
        else if (precisionStr == "MIXED")
        {
            precisionPolicy_ = PrecisionPolicy::MIXED;
        }
        else
        {
            FatalIOErrorInFunction(oglDict)
                << "Unknown precisionPolicy: " << precisionStr << nl
                << "Valid options: FP64, FP32, MIXED"
                << abort(FatalIOError);
        }

        iterativeRefinement_ = oglDict.lookupOrDefault<bool>
        (
            "iterativeRefinement",
            true
        );

        maxRefineIters_ = oglDict.lookupOrDefault<label>
        (
            "maxRefineIters",
            3
        );

        innerTolerance_ = oglDict.lookupOrDefault<scalar>
        (
            "innerTolerance",
            1e-4
        );

        cacheStructure_ = oglDict.lookupOrDefault<bool>
        (
            "cacheStructure",
            true
        );

        cacheValues_ = oglDict.lookupOrDefault<bool>
        (
            "cacheValues",
            false
        );

        debug_ = oglDict.lookupOrDefault<label>("debug", 0);
        timing_ = oglDict.lookupOrDefault<bool>("timing", false);

        // Adaptive precision settings
        useAdaptivePrecision_ = oglDict.lookupOrDefault<bool>
        (
            "useAdaptivePrecision",
            false
        );

        if (useAdaptivePrecision_ && oglDict.found("adaptivePrecision"))
        {
            adaptivePrecision_ = std::make_unique<AdaptivePrecision>
            (
                oglDict.subDict("adaptivePrecision")
            );
        }
        else if (useAdaptivePrecision_)
        {
            adaptivePrecision_ = std::make_unique<AdaptivePrecision>();
        }

        // Preconditioner type
        word precondStr = oglDict.lookupOrDefault<word>
        (
            "preconditioner",
            "Jacobi"
        );

        if (precondStr == "Jacobi")
        {
            preconditionerType_ = PreconditionerType::JACOBI;
        }
        else if (precondStr == "blockJacobi")
        {
            preconditionerType_ = PreconditionerType::BLOCK_JACOBI;
        }
        else if (precondStr == "ISAI")
        {
            preconditionerType_ = PreconditionerType::ISAI;
        }
        else if (precondStr == "blockJacobiISAI")
        {
            preconditionerType_ = PreconditionerType::BLOCK_JACOBI_ISAI;
        }
        else if (precondStr == "bjIsaiSandwich")
        {
            preconditionerType_ = PreconditionerType::BJ_ISAI_SANDWICH;
        }
        else if (precondStr == "bjIsaiAdditive")
        {
            preconditionerType_ = PreconditionerType::BJ_ISAI_ADDITIVE;
        }
        else if (precondStr == "bjIsaiGmres")
        {
            preconditionerType_ = PreconditionerType::BJ_ISAI_GMRES;
        }
        else if (precondStr == "bjIsaiInnerOuter")
        {
            preconditionerType_ = PreconditionerType::BJ_ISAI_INNER_OUTER;
        }
        else if (precondStr == "FFT")
        {
            preconditionerType_ = PreconditionerType::FFT;
        }
        else if (precondStr == "FFT_DIRECT")
        {
            preconditionerType_ = PreconditionerType::FFT_DIRECT;
        }
        else if (precondStr == "fftBlockJacobi")
        {
            preconditionerType_ = PreconditionerType::FFT_BLOCK_JACOBI;
        }
        else if (precondStr == "fftChebyshev")
        {
            preconditionerType_ = PreconditionerType::FFT_CHEBYSHEV;
        }
        else if (precondStr == "fftScaled")
        {
            preconditionerType_ = PreconditionerType::FFT_SCALED;
        }
        else if (precondStr == "fftBjVcycle")
        {
            preconditionerType_ = PreconditionerType::FFT_BJ_VCYCLE;
        }
        else if (precondStr == "geometricMgFft")
        {
            preconditionerType_ = PreconditionerType::GEOMETRIC_MG_FFT;
        }
        else if (precondStr == "multigrid")
        {
            preconditionerType_ = PreconditionerType::MULTIGRID;
        }
        else if (precondStr == "multigridFFT")
        {
            preconditionerType_ = PreconditionerType::MULTIGRID_FFT;
        }
        else if (precondStr == "ILU")
        {
            preconditionerType_ = PreconditionerType::ILU;
        }
        else
        {
            FatalIOErrorInFunction(oglDict)
                << "Unknown preconditioner: " << precondStr << nl
                << "Valid options: Jacobi, blockJacobi, ISAI, "
                << "blockJacobiISAI, bjIsaiSandwich, bjIsaiAdditive, "
                << "bjIsaiGmres, bjIsaiInnerOuter, FFT, fftBlockJacobi, "
                << "fftChebyshev, fftScaled, fftBjVcycle, "
                << "geometricMgFft, multigrid, multigridFFT, ILU"
                << abort(FatalIOError);
        }

        blockSize_ = oglDict.lookupOrDefault<label>("blockSize", 4);

        isaiSparsityPower_ = oglDict.lookupOrDefault<label>
        (
            "isaiSparsityPower",
            1
        );

        // FFT preconditioner grid dimensions (required for FFT types)
        if
        (
            preconditionerType_ == PreconditionerType::FFT
         || preconditionerType_ == PreconditionerType::FFT_DIRECT
         || preconditionerType_ == PreconditionerType::FFT_BLOCK_JACOBI
         || preconditionerType_ == PreconditionerType::FFT_CHEBYSHEV
         || preconditionerType_ == PreconditionerType::FFT_SCALED
         || preconditionerType_ == PreconditionerType::FFT_BJ_VCYCLE
         || preconditionerType_ == PreconditionerType::GEOMETRIC_MG_FFT
         || preconditionerType_ == PreconditionerType::MULTIGRID_FFT
        )
        {
            fftDimensions_ = oglDict.lookup<Vector<label>>("fftDimensions");
            meshSpacing_ = oglDict.lookup<Vector<scalar>>("meshSpacing");

            if (debug_ >= 1)
            {
                Info<< "    FFT grid: "
                    << fftDimensions_.x() << " x "
                    << fftDimensions_.y() << " x "
                    << fftDimensions_.z()
                    << ", spacing: ("
                    << meshSpacing_.x() << ", "
                    << meshSpacing_.y() << ", "
                    << meshSpacing_.z() << ")"
                    << endl;
            }
        }

        // V-cycle smoother settings
        if (preconditionerType_ == PreconditionerType::FFT_BJ_VCYCLE)
        {
            vcyclePreSmooth_ = oglDict.lookupOrDefault<label>
            (
                "vcyclePreSmooth", 1
            );
            vcyclePostSmooth_ = oglDict.lookupOrDefault<label>
            (
                "vcyclePostSmooth", 1
            );

            if (vcyclePreSmooth_ != vcyclePostSmooth_)
            {
                WarningInFunction
                    << "vcyclePreSmooth (" << vcyclePreSmooth_
                    << ") != vcyclePostSmooth (" << vcyclePostSmooth_
                    << "). Asymmetric V-cycle breaks CG symmetry."
                    << " Using GMRES would be required." << endl;
            }

            if (debug_ >= 1)
            {
                Info<< "    V-cycle: preSmooth=" << vcyclePreSmooth_
                    << " postSmooth=" << vcyclePostSmooth_ << endl;
            }
        }

        // Geometric MG + FFT settings
        if (preconditionerType_ == PreconditionerType::GEOMETRIC_MG_FFT)
        {
            vcyclePreSmooth_ = oglDict.lookupOrDefault<label>
            (
                "vcyclePreSmooth", 1
            );
            vcyclePostSmooth_ = oglDict.lookupOrDefault<label>
            (
                "vcyclePostSmooth", 1
            );
            vcycleChebDegree_ = oglDict.lookupOrDefault<label>
            (
                "vcycleChebDegree", 2
            );
            vcycleSmoother_ = oglDict.lookupOrDefault<word>
            (
                "vcycleSmoother", "chebyshev"
            );

            if (vcyclePreSmooth_ != vcyclePostSmooth_)
            {
                WarningInFunction
                    << "vcyclePreSmooth (" << vcyclePreSmooth_
                    << ") != vcyclePostSmooth (" << vcyclePostSmooth_
                    << "). Asymmetric V-cycle breaks CG symmetry."
                    << " Using GMRES would be required." << endl;
            }

            if (debug_ >= 1)
            {
                Info<< "    GeometricMG: preSmooth=" << vcyclePreSmooth_
                    << " postSmooth=" << vcyclePostSmooth_
                    << " smoother=" << vcycleSmoother_
                    << " chebDegree=" << vcycleChebDegree_ << endl;
            }
        }

        // Chebyshev-FFT preconditioner settings
        if (preconditionerType_ == PreconditionerType::FFT_CHEBYSHEV)
        {
            chebyDegree_ = oglDict.lookupOrDefault<label>
            (
                "chebyDegree", 3
            );
            chebyEigMin_ = oglDict.lookupOrDefault<scalar>
            (
                "chebyEigMin", 0.1
            );
            chebyEigMax_ = oglDict.lookupOrDefault<scalar>
            (
                "chebyEigMax", 4.0
            );

            if (debug_ >= 1)
            {
                Info<< "    Chebyshev-FFT: degree=" << chebyDegree_
                    << " foci=[" << chebyEigMin_ << ", "
                    << chebyEigMax_ << "]" << endl;
            }
        }

        // Multigrid preconditioner settings
        if
        (
            preconditionerType_ == PreconditionerType::MULTIGRID
         || preconditionerType_ == PreconditionerType::MULTIGRID_FFT
        )
        {
            mgMaxLevels_ = oglDict.lookupOrDefault<label>
            (
                "mgMaxLevels", 10
            );
            mgMinCoarseRows_ = oglDict.lookupOrDefault<label>
            (
                "mgMinCoarseRows", 64
            );
            mgSmootherIters_ = oglDict.lookupOrDefault<label>
            (
                "mgSmootherIters", 2
            );
            mgSmootherRelax_ = oglDict.lookupOrDefault<scalar>
            (
                "mgSmootherRelax", 0.9
            );
            mgSmoother_ = oglDict.lookupOrDefault<word>
            (
                "mgSmoother", "jacobi"
            );
            mgCacheInterval_ = oglDict.lookupOrDefault<label>
            (
                "mgCacheInterval", 10
            );
            mgCacheMaxIters_ = oglDict.lookupOrDefault<label>
            (
                "mgCacheMaxIters", 200
            );

            if (debug_ >= 1)
            {
                Info<< "    Multigrid: maxLevels=" << mgMaxLevels_
                    << " minCoarseRows=" << mgMinCoarseRows_
                    << " smoother=" << mgSmoother_
                    << " smootherIters=" << mgSmootherIters_
                    << " smootherRelax=" << mgSmootherRelax_
                    << endl;
            }
        }
    }
}


Foam::scalar Foam::OGL::OGLSolverBase::computeResidualNorm
(
    const scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    // Compute A*psi
    scalarField Apsi(psi.size());
    matrix_.Amul(Apsi, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // Compute residual = source - A*psi
    scalarField residual(source - Apsi);

    // Compute norm factor (same as in PCG.C)
    scalarField tmpField(psi.size());
    scalar normFactor = this->normFactor(psi, source, Apsi, tmpField);

    // Return normalized residual norm
    return gSumMag(residual, matrix().mesh().comm()) / normFactor;
}


Foam::solverPerformance Foam::OGL::OGLSolverBase::solveWithRefinement
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    // Setup performance tracking
    solverPerformance solverPerf(type() + ":FP32", fieldName_);

    const label nCells = psi.size();

    // Compute initial residual in FP64
    {
        ScopedTimer t(timer_, PerformanceTimer::Category::REFINEMENT_RESIDUAL);
        scalar residualNorm = computeResidualNorm(psi, source, cmpt);
        solverPerf.initialResidual() = residualNorm;
        solverPerf.finalResidual() = residualNorm;

        if (debug_ >= 1)
        {
            Info<< "OGL: Initial residual = " << residualNorm << endl;
        }

        // Check if already converged
        if (residualNorm < tolerance_)
        {
            return solverPerf;
        }
    }

    // Iterative refinement loop
    label totalIters = 0;

    for (label refine = 0; refine < maxRefineIters_; refine++)
    {
        // Compute FP64 residual: r = b - A*x
        scalarField Apsi(nCells);
        scalarField residual(nCells);
        {
            ScopedTimer t(timer_, PerformanceTimer::Category::REFINEMENT_RESIDUAL);
            matrix_.Amul(Apsi, psi, interfaceBouCoeffs_, interfaces_, cmpt);
            residual = source - Apsi;
        }

        // Solve correction in FP32: A*dx = r
        scalarField dx(nCells, 0.0);
        label iters;
        {
            ScopedTimer t(timer_, PerformanceTimer::Category::SOLVE_KERNEL);
            iters = solveFP32(dx, residual, innerTolerance_);
        }
        totalIters += iters;

        // Apply correction in FP64: x = x + dx
        forAll(psi, i)
        {
            psi[i] += dx[i];
        }

        // Compute new residual in FP64
        scalar residualNorm;
        {
            ScopedTimer t(timer_, PerformanceTimer::Category::REFINEMENT_RESIDUAL);
            residualNorm = computeResidualNorm(psi, source, cmpt);
        }
        solverPerf.finalResidual() = residualNorm;

        if (debug_ >= 1)
        {
            Info<< "OGL: Refinement " << refine + 1
                << ", iters = " << iters
                << ", residual = " << residualNorm << endl;
        }

        // Check convergence
        if (residualNorm < tolerance_)
        {
            break;
        }

        // Check relative convergence
        if
        (
            relTol_ > 0
         && residualNorm < relTol_ * solverPerf.initialResidual()
        )
        {
            break;
        }
    }

    solverPerf.nIterations() = totalIters;

    return solverPerf;
}


Foam::solverPerformance Foam::OGL::OGLSolverBase::solveWithAdaptivePrecision
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    // Setup performance tracking
    solverPerformance solverPerf(type() + ":adaptive", fieldName_);

    const label nCells = psi.size();

    // Reset adaptive precision for new solve
    adaptivePrecision_->reset();
    adaptivePrecision_->setTargetTolerance(tolerance_);

    // Compute initial residual in FP64
    scalar residualNorm;
    {
        ScopedTimer t(timer_, PerformanceTimer::Category::REFINEMENT_RESIDUAL);
        residualNorm = computeResidualNorm(psi, source, cmpt);
        solverPerf.initialResidual() = residualNorm;
        solverPerf.finalResidual() = residualNorm;

        if (debug_ >= 1)
        {
            Info<< "OGL Adaptive: Initial residual = " << residualNorm << endl;
        }

        // Check if already converged
        if (residualNorm < tolerance_)
        {
            return solverPerf;
        }
    }

    // Adaptive solve loop
    label totalIters = 0;
    label iteration = 0;

    for (label refine = 0; refine < maxRefineIters_; refine++)
    {
        // Compute FP64 residual: r = b - A*x
        scalarField Apsi(nCells);
        scalarField residual(nCells);
        {
            ScopedTimer t(timer_, PerformanceTimer::Category::REFINEMENT_RESIDUAL);
            matrix_.Amul(Apsi, psi, interfaceBouCoeffs_, interfaces_, cmpt);
            residual = source - Apsi;
        }

        // Get precision recommendation from adaptive controller
        AdaptivePrecision::Precision precision =
            adaptivePrecision_->update(residualNorm, iteration);

        // Solve correction in recommended precision
        scalarField dx(nCells, 0.0);
        label iters;
        {
            ScopedTimer t(timer_, PerformanceTimer::Category::SOLVE_KERNEL);

            if (precision == AdaptivePrecision::Precision::FP32)
            {
                iters = solveFP32(dx, residual, innerTolerance_);
            }
            else
            {
                iters = solveFP64(dx, residual, innerTolerance_);
            }
        }
        totalIters += iters;
        iteration++;

        // Apply correction in FP64: x = x + dx
        forAll(psi, i)
        {
            psi[i] += dx[i];
        }

        // Compute new residual in FP64
        {
            ScopedTimer t(timer_, PerformanceTimer::Category::REFINEMENT_RESIDUAL);
            residualNorm = computeResidualNorm(psi, source, cmpt);
        }
        solverPerf.finalResidual() = residualNorm;

        if (debug_ >= 1)
        {
            Info<< "OGL Adaptive: Refinement " << refine + 1
                << ", precision = "
                << (precision == AdaptivePrecision::Precision::FP32
                    ? "FP32" : "FP64")
                << ", iters = " << iters
                << ", residual = " << residualNorm << endl;
        }

        // Check convergence
        if (residualNorm < tolerance_)
        {
            break;
        }

        // Check relative convergence
        if
        (
            relTol_ > 0
         && residualNorm < relTol_ * solverPerf.initialResidual()
        )
        {
            break;
        }
    }

    solverPerf.nIterations() = totalIters;

    // Report adaptive precision statistics
    if (debug_ >= 1)
    {
        adaptivePrecision_->report();
    }

    return solverPerf;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::OGL::OGLSolverBase::OGLSolverBase
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    precisionPolicy_(PrecisionPolicy::FP32),
    iterativeRefinement_(true),
    maxRefineIters_(3),
    innerTolerance_(1e-4),
    cacheStructure_(true),
    cacheValues_(false),
    debug_(0),
    timing_(false),
    timer_(false),
    adaptivePrecision_(nullptr),
    useAdaptivePrecision_(false),
    preconditionerType_(PreconditionerType::JACOBI),
    blockSize_(4),
    isaiSparsityPower_(1),
    chebyDegree_(3),
    chebyEigMin_(0.1),
    chebyEigMax_(4.0),
    vcyclePreSmooth_(1),
    vcyclePostSmooth_(1),
    vcycleChebDegree_(2),
    vcycleSmoother_("chebyshev"),
    fftDimensions_(0, 0, 0),
    meshSpacing_(0, 0, 0),
    mgMaxLevels_(10),
    mgMinCoarseRows_(64),
    mgSmootherIters_(2),
    mgSmootherRelax_(0.9),
    mgSmoother_("jacobi"),
    mgCacheInterval_(10),
    mgCacheMaxIters_(200)
{
    readOGLControls();

    // Initialize timer with timing flag
    timer_.setEnabled(timing_);

    // Configure executor before first use (only has effect if not yet initialized)
    if (!OGLExecutor::initialized())
    {
        dictionary defaultDict;
        defaultDict.add("debug", debug_);
        OGLExecutor::configure(defaultDict);
    }
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

Foam::solverPerformance Foam::OGL::OGLSolverBase::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    // Start total solve timer
    ScopedTimer totalTimer(timer_, PerformanceTimer::Category::TOTAL_SOLVE);

    solverPerformance solverPerf;

    // Choose solve path based on precision policy
    switch (precisionPolicy_)
    {
        case PrecisionPolicy::FP64:
        {
            // Pure FP64 solve
            solverPerf = solverPerformance(type() + ":FP64", fieldName_);

            scalar initResidual = computeResidualNorm(psi, source, cmpt);
            solverPerf.initialResidual() = initResidual;

            if (initResidual >= tolerance_)
            {
                // Compute an effective relative reduction factor for Ginkgo.
                // Ginkgo's ResidualNorm criterion is relative:
                //   ||r_k||/||r_0|| < factor
                // When relTol > 0, use it directly (matches OpenFOAM's relTol).
                // When relTol = 0 (e.g. pFinal), convert the absolute
                // tolerance to an equivalent reduction factor:
                //   factor = tolerance / initResidual
                // so that ||r_k|| ~ tolerance * normFactor (approximately
                // matching OpenFOAM's absolute convergence level).
                scalar effectiveRelTol;
                if (relTol_ > 0)
                {
                    effectiveRelTol = relTol_;
                }
                else
                {
                    effectiveRelTol = tolerance_ / initResidual;
                    effectiveRelTol = max(effectiveRelTol, scalar(1e-8));
                    effectiveRelTol = min(effectiveRelTol, scalar(1.0));
                }

                ScopedTimer t(timer_, PerformanceTimer::Category::SOLVE_KERNEL);
                label iters = solveFP64(psi, source, effectiveRelTol);
                solverPerf.nIterations() = iters;
            }

            solverPerf.finalResidual() = computeResidualNorm(psi, source, cmpt);
            break;
        }

        case PrecisionPolicy::FP32:
        case PrecisionPolicy::MIXED:
        {
            if (useAdaptivePrecision_ && adaptivePrecision_)
            {
                // Use adaptive precision switching
                solverPerf = solveWithAdaptivePrecision(psi, source, cmpt);
            }
            else if (iterativeRefinement_)
            {
                solverPerf = solveWithRefinement(psi, source, cmpt);
            }
            else
            {
                // FP32 without refinement (not recommended for production)
                solverPerf = solverPerformance(type() + ":FP32", fieldName_);

                scalar initResidual = computeResidualNorm(psi, source, cmpt);
                solverPerf.initialResidual() = initResidual;

                if (initResidual >= tolerance_)
                {
                    ScopedTimer t(timer_, PerformanceTimer::Category::SOLVE_KERNEL);
                    label iters = solveFP32(psi, source, tolerance_);
                    solverPerf.nIterations() = iters;
                }

                solverPerf.finalResidual() =
                    computeResidualNorm(psi, source, cmpt);
            }
            break;
        }
    }

    return solverPerf;
}


void Foam::OGL::OGLSolverBase::reportPerformance() const
{
    if (timing_)
    {
        timer_.report(fieldName_);
    }
}


// ************************************************************************* //
