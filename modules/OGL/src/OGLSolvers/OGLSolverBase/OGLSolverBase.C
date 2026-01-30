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
    solverPerformance solverPerf(typeName + ":FP32", fieldName_);

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
    timer_(false)
{
    readOGLControls();

    // Initialize timer with timing flag
    timer_.setEnabled(timing_);

    // Ensure executor is initialized
    if (!OGLExecutor::initialized())
    {
        dictionary defaultDict;
        defaultDict.add("debug", debug_);
        OGLExecutor::initialize(defaultDict);
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
            solverPerf = solverPerformance(typeName + ":FP64", fieldName_);

            scalar initResidual = computeResidualNorm(psi, source, cmpt);
            solverPerf.initialResidual() = initResidual;

            if (initResidual >= tolerance_)
            {
                ScopedTimer t(timer_, PerformanceTimer::Category::SOLVE_KERNEL);
                label iters = solveFP64(psi, source, tolerance_);
                solverPerf.nIterations() = iters;
            }

            solverPerf.finalResidual() = computeResidualNorm(psi, source, cmpt);
            break;
        }

        case PrecisionPolicy::FP32:
        case PrecisionPolicy::MIXED:
        {
            if (iterativeRefinement_)
            {
                solverPerf = solveWithRefinement(psi, source, cmpt);
            }
            else
            {
                // FP32 without refinement (not recommended for production)
                solverPerf = solverPerformance(typeName + ":FP32", fieldName_);

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
