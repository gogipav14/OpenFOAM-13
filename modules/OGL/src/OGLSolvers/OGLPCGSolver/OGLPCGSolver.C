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

void Foam::OGL::OGLPCGSolver::updateSolverF32() const
{
    auto exec = OGLExecutor::instance().executor();

    // Create the linear operator
    operatorF32_ = std::make_shared<FoamGinkgoLinOp<float>>
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
    auto precond = gko::preconditioner::Jacobi<float, int>::build()
        .with_max_block_size(1u)
        .on(exec);

    // Create CG solver factory
    auto solverFactory = gko::solver::Cg<float>::build()
        .with_preconditioner(precond)
        .with_criteria(
            gko::stop::Iteration::build()
                .with_max_iters(static_cast<gko::size_type>(maxIter_))
                .on(exec),
            gko::stop::ResidualNorm<float>::build()
                .with_reduction_factor(static_cast<float>(innerTolerance_))
                .on(exec)
        )
        .on(exec);

    // Generate solver from operator
    solverF32_ = solverFactory->generate(operatorF32_);
}


void Foam::OGL::OGLPCGSolver::updateSolverF64() const
{
    auto exec = OGLExecutor::instance().executor();

    // Create the linear operator
    operatorF64_ = std::make_shared<FoamGinkgoLinOp<double>>
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
    auto precond = gko::preconditioner::Jacobi<double, int>::build()
        .with_max_block_size(1u)
        .on(exec);

    // Create CG solver factory
    auto solverFactory = gko::solver::Cg<double>::build()
        .with_preconditioner(precond)
        .with_criteria(
            gko::stop::Iteration::build()
                .with_max_iters(static_cast<gko::size_type>(maxIter_))
                .on(exec),
            gko::stop::ResidualNorm<double>::build()
                .with_reduction_factor(tolerance_)
                .on(exec)
        )
        .on(exec);

    // Generate solver from operator
    solverF64_ = solverFactory->generate(operatorF64_);
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

Foam::label Foam::OGL::OGLPCGSolver::solveFP32
(
    scalarField& psi,
    const scalarField& source,
    const scalar tolerance
) const
{
    auto exec = OGLExecutor::instance().executor();

    // Update solver if needed
    if (!solverF32_)
    {
        updateSolverF32();
    }
    else if (!cacheValues_)
    {
        // Invalidate values to force update
        operatorF32_->invalidateValues();
    }

    // Convert to Ginkgo vectors
    auto b = FP32CastWrapper::toGinkgoF32(exec, source);
    auto x = FP32CastWrapper::toGinkgoF32(exec, psi);

    // Create iteration logger to get iteration count
    auto logger = gko::log::Convergence<float>::create();
    solverF32_->add_logger(logger);

    // Solve
    solverF32_->apply(b.get(), x.get());

    // Get iteration count
    label numIters = static_cast<label>(logger->get_num_iterations());

    // Remove logger
    solverF32_->remove_logger(logger.get());

    // Copy solution back
    FP32CastWrapper::fromGinkgoF32(x.get(), psi);

    return numIters;
}


Foam::label Foam::OGL::OGLPCGSolver::solveFP64
(
    scalarField& psi,
    const scalarField& source,
    const scalar tolerance
) const
{
    auto exec = OGLExecutor::instance().executor();

    // Update solver if needed
    if (!solverF64_)
    {
        updateSolverF64();
    }
    else if (!cacheValues_)
    {
        // Invalidate values to force update
        operatorF64_->invalidateValues();
    }

    // Convert to Ginkgo vectors
    auto b = FP32CastWrapper::toGinkgoF64(exec, source);
    auto x = FP32CastWrapper::toGinkgoF64(exec, psi);

    // Create iteration logger to get iteration count
    auto logger = gko::log::Convergence<double>::create();
    solverF64_->add_logger(logger);

    // Solve
    solverF64_->apply(b.get(), x.get());

    // Get iteration count
    label numIters = static_cast<label>(logger->get_num_iterations());

    // Remove logger
    solverF64_->remove_logger(logger.get());

    // Copy solution back
    FP32CastWrapper::fromGinkgoF64(x.get(), psi);

    return numIters;
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
