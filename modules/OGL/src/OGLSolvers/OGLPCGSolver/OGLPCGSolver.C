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

#include "OGLPCGSolver.H"
#include "OGLExecutor.H"
#include "FP32CastWrapper.H"
#include "addToRunTimeSelectionTable.H"

#include <cmath>
#include <limits>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{
    defineTypeNameAndDebug(OGLPCGSolver, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<OGLPCGSolver>
        addOGLPCGSymMatrixConstructorToTable_;
}
}


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<typename ValueType>
void Foam::OGL::OGLPCGSolver::updateSolverImpl
(
    std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op,
    std::shared_ptr<gko::solver::Cg<ValueType>>& solver,
    ValueType tolerance
) const
{
    auto exec = OGLExecutor::instance().executor();

    try
    {
        // Create the linear operator
        op = std::make_shared<FoamGinkgoLinOp<ValueType>>
        (
            exec,
            matrix_,
            interfaceBouCoeffs_,
            interfaceIntCoeffs_,
            interfaces_,
            0,  // cmpt
            true,  // includeInterfaces
            cacheStructure_,
            cacheValues_
        );

        // Create Jacobi preconditioner
        auto precond = gko::preconditioner::Jacobi<ValueType, int>::build()
            .with_max_block_size(1u)
            .on(exec);

        // Create CG solver factory
        auto solverFactory = gko::solver::Cg<ValueType>::build()
            .with_preconditioner(precond)
            .with_criteria(
                gko::stop::Iteration::build()
                    .with_max_iters(static_cast<gko::size_type>(maxIter_))
                    .on(exec),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(tolerance)
                    .on(exec)
            )
            .on(exec);

        // Generate solver from operator
        solver = solverFactory->generate(op);
    }
    catch (const std::exception& e)
    {
        OGLExecutor::checkGinkgoError("updateSolverImpl", e);
    }
}


template<typename ValueType>
Foam::label Foam::OGL::OGLPCGSolver::solveImpl
(
    scalarField& psi,
    const scalarField& source,
    std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op,
    std::shared_ptr<gko::solver::Cg<ValueType>>& solver,
    ValueType tolerance
) const
{
    auto exec = OGLExecutor::instance().executor();
    label numIters = 0;

    try
    {
        // Update solver if needed
        if (!solver)
        {
            updateSolverImpl(op, solver, tolerance);
        }
        else if (!cacheValues_)
        {
            // Invalidate values to force update
            op->invalidateValues();
        }

        // Convert to Ginkgo vectors using appropriate precision
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

        // Create iteration logger to get iteration count
        auto logger = gko::log::Convergence<ValueType>::create();
        solver->add_logger(logger);

        // Solve with error handling
        solver->apply(b.get(), x.get());

        // Synchronize to ensure solve completed
        OGLExecutor::instance().synchronize();

        // Get iteration count
        numIters = static_cast<label>(logger->get_num_iterations());

        // Check if solver converged
        if (logger->has_converged())
        {
            if (debug_ >= 2)
            {
                Info<< "OGLPCGSolver: Converged in " << numIters
                    << " iterations" << endl;
            }
        }
        else
        {
            if (debug_ >= 1)
            {
                WarningInFunction
                    << "OGLPCGSolver: Did not converge in " << numIters
                    << " iterations (max: " << maxIter_ << ")" << endl;
            }
        }

        // Remove logger
        solver->remove_logger(logger.get());

        // Copy solution back
        if constexpr (std::is_same<ValueType, float>::value)
        {
            FP32CastWrapper::fromGinkgoF32(x.get(), psi);
        }
        else
        {
            FP32CastWrapper::fromGinkgoF64(x.get(), psi);
        }

        // Numerical safety: check for NaN/Inf in solution
        bool hasNaN = false;
        bool hasInf = false;
        forAll(psi, i)
        {
            if (std::isnan(psi[i]))
            {
                hasNaN = true;
                break;
            }
            if (std::isinf(psi[i]))
            {
                hasInf = true;
                break;
            }
        }

        if (hasNaN)
        {
            FatalErrorInFunction
                << "OGLPCGSolver: NaN detected in solution"
                << abort(FatalError);
        }

        if (hasInf)
        {
            WarningInFunction
                << "OGLPCGSolver: Infinity detected in solution" << endl;
        }
    }
    catch (const std::exception& e)
    {
        OGLExecutor::checkGinkgoError("solveImpl", e);
    }

    return numIters;
}


void Foam::OGL::OGLPCGSolver::updateSolverF32() const
{
    updateSolverImpl<float>
    (
        operatorF32_,
        solverF32_,
        static_cast<float>(innerTolerance_)
    );
}


void Foam::OGL::OGLPCGSolver::updateSolverF64() const
{
    updateSolverImpl<double>(operatorF64_, solverF64_, tolerance_);
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

Foam::label Foam::OGL::OGLPCGSolver::solveFP32
(
    scalarField& psi,
    const scalarField& source,
    const scalar tolerance
) const
{
    return solveImpl<float>
    (
        psi,
        source,
        operatorF32_,
        solverF32_,
        static_cast<float>(innerTolerance_)
    );
}


Foam::label Foam::OGL::OGLPCGSolver::solveFP64
(
    scalarField& psi,
    const scalarField& source,
    const scalar tolerance
) const
{
    return solveImpl<double>(psi, source, operatorF64_, solverF64_, tolerance_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::OGL::OGLPCGSolver::OGLPCGSolver
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
    solverF32_(nullptr),
    solverF64_(nullptr)
{
    if (debug_ >= 1)
    {
        Info<< "OGLPCGSolver: Created for field " << fieldName << nl
            << "  precisionPolicy: "
            << (precisionPolicy_ == PrecisionPolicy::FP64 ? "FP64" :
                precisionPolicy_ == PrecisionPolicy::FP32 ? "FP32" : "MIXED")
            << nl
            << "  iterativeRefinement: " << iterativeRefinement_ << nl
            << "  maxRefineIters: " << maxRefineIters_ << nl
            << "  innerTolerance: " << innerTolerance_ << nl
            << "  cacheStructure: " << cacheStructure_ << nl
            << "  cacheValues: " << cacheValues_ << nl
            << endl;
    }
}


// ************************************************************************* //
