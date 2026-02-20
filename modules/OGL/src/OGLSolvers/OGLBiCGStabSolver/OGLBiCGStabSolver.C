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

#include "OGLBiCGStabSolver.H"
#include "OGLExecutor.H"
#include "FP32CastWrapper.H"
#include "addToRunTimeSelectionTable.H"

#include <chrono>
#include <cmath>
#include <limits>
#include <ginkgo/core/factorization/par_ilu.hpp>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{
    defineTypeNameAndDebug(OGLBiCGStabSolver, 0);

    lduMatrix::solver::addasymMatrixConstructorToTable<OGLBiCGStabSolver>
        addOGLBiCGStabAsymMatrixConstructorToTable_;
}
}

// Static cache members
std::map<Foam::word, Foam::OGL::OGLBiCGStabSolver::SolverCache>
    Foam::OGL::OGLBiCGStabSolver::cache_;
std::mutex Foam::OGL::OGLBiCGStabSolver::cacheMutex_;


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<typename ValueType>
void Foam::OGL::OGLBiCGStabSolver::updateSolverImpl
(
    std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op,
    std::shared_ptr<gko::LinOp>& solver,
    ValueType tolerance
) const
{
    auto exec = OGLExecutor::instance().executor();

    try
    {
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

            if (debug_ >= 2)
            {
                Info<< "OGLBiCGStabSolver: Reusing cached operator" << endl;
            }
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
                0,  // cmpt
                true,  // includeInterfaces
                cacheStructure_,
                cacheValues_
            );
        }

        // Stopping criteria (shared by all preconditioner paths)
        auto iterCrit = gko::share(
            gko::stop::Iteration::build()
                .with_max_iters(static_cast<gko::size_type>(maxIter_))
                .on(exec)
        );
        auto resCrit = gko::share(
            gko::stop::ResidualNorm<ValueType>::build()
                .with_reduction_factor(tolerance)
                .with_baseline(gko::stop::mode::initial_resnorm)
                .on(exec)
        );

        // Helper lambdas: build preconditioners from CSR matrix
        auto makeBJ = [&]() -> std::shared_ptr<const gko::LinOp>
        {
            return gko::preconditioner::Jacobi<ValueType, int>::build()
                .with_max_block_size(static_cast<unsigned>(blockSize_))
                .on(exec)
                ->generate(op->localMatrix());
        };

        // Diagonal Jacobi as lightweight ISAI substitute
        // (GeneralIsai may not be available in Ginkgo v1.11.0)
        auto makeISAI = [&]() -> std::shared_ptr<const gko::LinOp>
        {
            return gko::preconditioner::Jacobi<ValueType, int>::build()
                .with_max_block_size(1u)
                .on(exec)
                ->generate(op->localMatrix());
        };

        // Preconditioner and solver creation.
        // Only asymmetric-compatible preconditioner paths are included.
        // SPD-only (sandwich, additive, FFT) and multigrid are excluded.

        if (preconditionerType_ == PreconditionerType::BLOCK_JACOBI_ISAI)
        {
            // Multiplicative: z = ISAI(BJ(r)) — BiCGStab
            auto bj = makeBJ();
            auto isai = makeISAI();
            std::vector<std::shared_ptr<const gko::LinOp>> ops =
                {isai, bj};
            auto composed = gko::share(
                gko::Composition<ValueType>::create(
                    ops.begin(), ops.end()
                )
            );

            solver = gko::solver::Bicgstab<ValueType>::build()
                .with_generated_preconditioner(composed)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());
        }
        else if
        (
            preconditionerType_ == PreconditionerType::BJ_ISAI_GMRES
        )
        {
            // Multiplicative with GMRES (no SPD requirement)
            auto bj = makeBJ();
            auto isai = makeISAI();
            std::vector<std::shared_ptr<const gko::LinOp>> ops =
                {isai, bj};
            auto composed = gko::share(
                gko::Composition<ValueType>::create(
                    ops.begin(), ops.end()
                )
            );

            solver = gko::solver::Gmres<ValueType>::build()
                .with_krylov_dim(100u)
                .with_generated_preconditioner(composed)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());
        }
        else if (preconditionerType_ == PreconditionerType::ILU)
        {
            // ParILU factorization — computes L,U factors from system matrix
            auto parIluFact = gko::share(
                gko::factorization::ParIlu<ValueType, int>::build()
                    .on(exec)
            );

            // ILU preconditioner with ISAI triangular solves (parallel SpMV)
            // ISAI precomputes approximate inverses of L and U, turning
            // sequential forward/backward substitution into parallel SpMV.
            // (LowerTrs/UpperTrs were ~2.5x slower due to sequential nature)
            auto iluFactory = gko::share(
                gko::preconditioner::Ilu<
                    gko::preconditioner::LowerIsai<ValueType, int>,
                    gko::preconditioner::UpperIsai<ValueType, int>,
                    false,
                    int
                >::build()
                    .with_factorization(parIluFact)
                    .on(exec)
            );

            solver = gko::solver::Bicgstab<ValueType>::build()
                .with_preconditioner(iluFactory)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());
        }
        else
        {
            // Factory-based preconditioner path (Jacobi / Block Jacobi)
            std::shared_ptr<gko::LinOpFactory> precond;

            switch (preconditionerType_)
            {
                case PreconditionerType::BLOCK_JACOBI:
                    precond = gko::share(
                        gko::preconditioner::Jacobi<ValueType, int>::build()
                            .with_max_block_size(
                                static_cast<unsigned>(blockSize_)
                            )
                            .on(exec)
                    );
                    break;

                case PreconditionerType::JACOBI:
                default:
                    precond = gko::share(
                        gko::preconditioner::Jacobi<ValueType, int>::build()
                            .with_max_block_size(1u)
                            .on(exec)
                    );
                    break;
            }

            solver = gko::solver::Bicgstab<ValueType>::build()
                .with_preconditioner(precond)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());
        }
    }
    catch (const std::exception& e)
    {
        OGLExecutor::checkGinkgoError("OGLBiCGStabSolver::updateSolverImpl", e);
    }
}


template<typename ValueType>
Foam::label Foam::OGL::OGLBiCGStabSolver::solveImpl
(
    scalarField& psi,
    const scalarField& source,
    std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op,
    std::shared_ptr<gko::LinOp>& solver,
    ValueType tolerance
) const
{
    auto exec = OGLExecutor::instance().executor();
    label numIters = 0;

    try
    {
        // Always rebuild solver — momentum matrix changes every PISO
        // corrector.  Only the operator (GPU CSR structure) is cached.
        if (op)
        {
            op->updatePointers
            (
                matrix_,
                interfaceBouCoeffs_,
                interfaceIntCoeffs_,
                interfaces_,
                0
            );

            if (!cacheValues_)
            {
                op->invalidateValues();
            }
        }

        updateSolverImpl(op, solver, tolerance);

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
        auto logger = gko::share(gko::log::Convergence<ValueType>::create());
        solver->add_logger(logger);

        // Solve
        exec->synchronize();
        auto solveStart = std::chrono::high_resolution_clock::now();

        solver->apply(b.get(), x.get());

        OGLExecutor::instance().synchronize();

        auto solveEnd = std::chrono::high_resolution_clock::now();
        double solveMs = std::chrono::duration<double, std::milli>(
            solveEnd - solveStart
        ).count();

        // Get iteration count
        numIters = static_cast<label>(logger->get_num_iterations());

        if (debug_ >= 1)
        {
            Info<< "OGLBiCGStabSolver: BiCGStab solve: " << solveMs << " ms, "
                << numIters << " iters, "
                << (numIters > 0 ? solveMs / numIters : 0)
                << " ms/iter" << endl;
        }

        // Check convergence
        if (!logger->has_converged())
        {
            if (debug_ >= 1)
            {
                WarningInFunction
                    << "OGLBiCGStabSolver: Did not converge in " << numIters
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
                << "OGLBiCGStabSolver: NaN detected in solution"
                << abort(FatalError);
        }

        if (hasInf)
        {
            WarningInFunction
                << "OGLBiCGStabSolver: Infinity detected in solution" << endl;
        }
    }
    catch (const std::exception& e)
    {
        OGLExecutor::checkGinkgoError("OGLBiCGStabSolver::solveImpl", e);
    }

    return numIters;
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

Foam::label Foam::OGL::OGLBiCGStabSolver::solveFP32
(
    scalarField& psi,
    const scalarField& source,
    const scalar tolerance
) const
{
    label iters = solveImpl<float>
    (
        psi,
        source,
        operatorF32_,
        solverF32_,
        static_cast<float>(innerTolerance_)
    );

    // Persist operator to static cache (structure reuse across instantiations)
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[fieldName_].opF32 = operatorF32_;
    }

    return iters;
}


Foam::label Foam::OGL::OGLBiCGStabSolver::solveFP64
(
    scalarField& psi,
    const scalarField& source,
    const scalar tolerance
) const
{
    // tolerance parameter is the effective Ginkgo reduction factor,
    // computed by OGLSolverBase::solve() from relTol_ or tolerance_/initResidual
    label iters = solveImpl<double>(
        psi, source, operatorF64_, solverF64_, tolerance
    );

    // Persist operator to static cache (structure reuse across instantiations)
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[fieldName_].opF64 = operatorF64_;
    }

    return iters;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::OGL::OGLBiCGStabSolver::OGLBiCGStabSolver
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
    // Restore cached operator from previous instantiation (if any).
    // Only the operator (GPU CSR structure) is cached — the solver
    // is rebuilt each call since momentum matrices change frequently.
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto it = cache_.find(fieldName);
        if (it != cache_.end())
        {
            operatorF32_ = it->second.opF32;
            operatorF64_ = it->second.opF64;

            if (debug_ >= 2)
            {
                Info<< "OGLBiCGStabSolver: Restored cached operator"
                    << " for " << fieldName << endl;
            }
        }
    }

    if (debug_ >= 1)
    {
        Info<< "OGLBiCGStabSolver: Created for field " << fieldName << nl
            << "  precisionPolicy: "
            << (precisionPolicy_ == PrecisionPolicy::FP64 ? "FP64" :
                precisionPolicy_ == PrecisionPolicy::FP32 ? "FP32" : "MIXED")
            << nl
            << "  preconditioner: "
            << [&]() -> const char* {
                switch (preconditionerType_)
                {
                    case PreconditionerType::JACOBI: return "Jacobi";
                    case PreconditionerType::BLOCK_JACOBI: return "blockJacobi";
                    case PreconditionerType::BLOCK_JACOBI_ISAI:
                        return "blockJacobiISAI";
                    case PreconditionerType::BJ_ISAI_GMRES:
                        return "bjIsaiGmres";
                    default: return "unknown (falling back to Jacobi)";
                }
            }()
            << nl
            << "  blockSize: " << blockSize_
            << nl
            << endl;
    }
}


// ************************************************************************* //
