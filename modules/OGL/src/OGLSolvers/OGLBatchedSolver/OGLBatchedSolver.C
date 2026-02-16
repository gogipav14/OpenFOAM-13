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

#include "OGLBatchedSolver.H"
#include "OGLExecutor.H"
#include "FP32CastWrapper.H"
#include "AdditiveLinOp.H"
#include "FFTPreconditioner.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{
    defineTypeNameAndDebug(OGLBatchedSolver, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<OGLBatchedSolver>
        addOGLBatchedSymMatrixConstructorToTable_;
}
}


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::OGL::OGLBatchedSolver::initBatch(label nCells) const
{
    batchedRHS_.resize(batchSize_);
    batchedSolution_.resize(batchSize_);

    for (label i = 0; i < batchSize_; i++)
    {
        batchedRHS_[i].setSize(nCells, 0.0);
        batchedSolution_[i].setSize(nCells, 0.0);
    }
}


Foam::label Foam::OGL::OGLBatchedSolver::executeBatchedSolve() const
{
    // TODO: Implement true batched solve using Ginkgo batched solvers
    // For now, fall back to sequential
    return executeSequentialSolve();
}


Foam::label Foam::OGL::OGLBatchedSolver::executeSequentialSolve() const
{
    label totalIterations = 0;

    for (label i = 0; i < currentBatchIndex_; i++)
    {
        // Use parent class solve for each component
        if (precisionPolicy_ == PrecisionPolicy::FP32 ||
            precisionPolicy_ == PrecisionPolicy::MIXED)
        {
            totalIterations += solveFP32
            (
                batchedSolution_[i],
                batchedRHS_[i],
                tolerance_
            );
        }
        else
        {
            totalIterations += solveFP64
            (
                batchedSolution_[i],
                batchedRHS_[i],
                tolerance_
            );
        }
    }

    return totalIterations;
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

Foam::label Foam::OGL::OGLBatchedSolver::solveFP32
(
    scalarField& psi,
    const scalarField& source,
    const scalar tolerance
) const
{
    auto exec = OGLExecutor::instance().executor();

    // Create operator if needed
    if (!operatorF32_)
    {
        operatorF32_ = std::make_shared<FoamGinkgoLinOp<float>>
        (
            exec,
            matrix_,
            interfaceBouCoeffs_,
            interfaceIntCoeffs_,
            interfaces_,
            0,
            true,
            cacheStructure_,
            cacheValues_
        );
    }

    // Convert to Ginkgo vectors
    auto b = FP32CastWrapper::toGinkgoF32(exec, source);
    auto x = FP32CastWrapper::toGinkgoF32(exec, psi);

    // Stopping criteria
    auto iterCrit = gko::share(
        gko::stop::Iteration::build()
            .with_max_iters(static_cast<gko::size_type>(maxIter_))
            .on(exec)
    );
    auto resCrit = gko::share(
        gko::stop::ResidualNorm<float>::build()
            .with_reduction_factor(static_cast<float>(tolerance))
            .on(exec)
    );

    // Helper lambdas for composite preconditioners
    auto makeBJ = [&]() -> std::shared_ptr<const gko::LinOp>
    {
        return gko::preconditioner::Jacobi<float, int>::build()
            .with_max_block_size(static_cast<unsigned>(blockSize_))
            .on(exec)
            ->generate(operatorF32_->localMatrix());
    };

    auto makeISAI = [&]() -> std::shared_ptr<const gko::LinOp>
    {
        return gko::preconditioner::SpdIsai<float, int>::build()
            .with_sparsity_power(isaiSparsityPower_)
            .on(exec)
            ->generate(operatorF32_->localMatrix());
    };

    // Create preconditioner and solver
    std::unique_ptr<gko::LinOp> solver;

    if (preconditionerType_ == PreconditionerType::BLOCK_JACOBI_ISAI)
    {
        auto bj = makeBJ();
        auto isai = makeISAI();
        std::vector<std::shared_ptr<const gko::LinOp>> ops = {isai, bj};
        auto composed = gko::share(
            gko::Composition<float>::create(ops.begin(), ops.end())
        );
        solver = gko::solver::Cg<float>::build()
            .with_generated_preconditioner(composed)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF32_->localMatrix());
    }
    else if (preconditionerType_ == PreconditionerType::BJ_ISAI_SANDWICH)
    {
        auto bj = makeBJ();
        auto isai = makeISAI();
        std::vector<std::shared_ptr<const gko::LinOp>> ops = {bj, isai, bj};
        auto composed = gko::share(
            gko::Composition<float>::create(ops.begin(), ops.end())
        );
        solver = gko::solver::Cg<float>::build()
            .with_generated_preconditioner(composed)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF32_->localMatrix());
    }
    else if (preconditionerType_ == PreconditionerType::BJ_ISAI_ADDITIVE)
    {
        auto bj = makeBJ();
        auto isai = makeISAI();
        auto additive = gko::share(
            AdditiveLinOp<float>::create(bj, isai)
        );
        solver = gko::solver::Cg<float>::build()
            .with_generated_preconditioner(additive)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF32_->localMatrix());
    }
    else if (preconditionerType_ == PreconditionerType::BJ_ISAI_GMRES)
    {
        auto bj = makeBJ();
        auto isai = makeISAI();
        std::vector<std::shared_ptr<const gko::LinOp>> ops = {isai, bj};
        auto composed = gko::share(
            gko::Composition<float>::create(ops.begin(), ops.end())
        );
        solver = gko::solver::Gmres<float>::build()
            .with_krylov_dim(100u)
            .with_generated_preconditioner(composed)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF32_->localMatrix());
    }
    else if (preconditionerType_ == PreconditionerType::BJ_ISAI_INNER_OUTER)
    {
        auto bjFactory = gko::share(
            gko::preconditioner::Jacobi<float, int>::build()
                .with_max_block_size(static_cast<unsigned>(blockSize_))
                .on(exec)
        );
        auto innerCrit = gko::share(
            gko::stop::Iteration::build()
                .with_max_iters(static_cast<gko::size_type>(5))
                .on(exec)
        );
        auto innerSolver = gko::share(
            gko::solver::Cg<float>::build()
                .with_preconditioner(bjFactory)
                .with_criteria(innerCrit)
                .on(exec)
        );
        solver = gko::solver::Gmres<float>::build()
            .with_krylov_dim(100u)
            .with_flexible(true)
            .with_preconditioner(innerSolver)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF32_->localMatrix());
    }
    else if (preconditionerType_ == PreconditionerType::FFT)
    {
        auto nCells = static_cast<gko::size_type>(
            fftDimensions_.x() * fftDimensions_.y() * fftDimensions_.z()
        );
        auto fftPrecond = gko::share(
            FFTPreconditioner<float>::create(
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

        solver = gko::solver::Cg<float>::build()
            .with_generated_preconditioner(fftPrecond)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF32_->localMatrix());
    }
    else if (preconditionerType_ == PreconditionerType::FFT_BLOCK_JACOBI)
    {
        auto nCells = static_cast<gko::size_type>(
            fftDimensions_.x() * fftDimensions_.y() * fftDimensions_.z()
        );
        auto fft = gko::share(
            FFTPreconditioner<float>::create(
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
        auto bj = makeBJ();
        auto additive = gko::share(
            AdditiveLinOp<float>::create(fft, bj)
        );

        solver = gko::solver::Cg<float>::build()
            .with_generated_preconditioner(additive)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF32_->localMatrix());
    }
    else
    {
        std::shared_ptr<gko::LinOpFactory> precond;
        switch (preconditionerType_)
        {
            case PreconditionerType::BLOCK_JACOBI:
                precond = gko::share(
                    gko::preconditioner::Jacobi<float, int>::build()
                        .with_max_block_size(
                            static_cast<unsigned>(blockSize_)
                        )
                        .on(exec)
                );
                break;
            case PreconditionerType::ISAI:
                precond = gko::share(
                    gko::preconditioner::SpdIsai<float, int>::build()
                        .with_sparsity_power(isaiSparsityPower_)
                        .on(exec)
                );
                break;
            case PreconditionerType::JACOBI:
            default:
                precond = gko::share(
                    gko::preconditioner::Jacobi<float, int>::build()
                        .with_max_block_size(1u)
                        .on(exec)
                );
                break;
        }
        solver = gko::solver::Cg<float>::build()
            .with_preconditioner(precond)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF32_->localMatrix());
    }

    auto logger = gko::share(gko::log::Convergence<float>::create());
    solver->add_logger(logger);

    solver->apply(b.get(), x.get());

    label numIters = static_cast<label>(logger->get_num_iterations());

    // Copy solution back
    FP32CastWrapper::fromGinkgoF32(x.get(), psi);

    return numIters;
}


Foam::label Foam::OGL::OGLBatchedSolver::solveFP64
(
    scalarField& psi,
    const scalarField& source,
    const scalar tolerance
) const
{
    auto exec = OGLExecutor::instance().executor();

    // Create operator if needed
    if (!operatorF64_)
    {
        operatorF64_ = std::make_shared<FoamGinkgoLinOp<double>>
        (
            exec,
            matrix_,
            interfaceBouCoeffs_,
            interfaceIntCoeffs_,
            interfaces_,
            0,
            true,
            cacheStructure_,
            cacheValues_
        );
    }

    // Convert to Ginkgo vectors
    auto b = FP32CastWrapper::toGinkgoF64(exec, source);
    auto x = FP32CastWrapper::toGinkgoF64(exec, psi);

    // Stopping criteria
    auto iterCrit = gko::share(
        gko::stop::Iteration::build()
            .with_max_iters(static_cast<gko::size_type>(maxIter_))
            .on(exec)
    );
    auto resCrit = gko::share(
        gko::stop::ResidualNorm<double>::build()
            .with_reduction_factor(tolerance)
            .on(exec)
    );

    // Helper lambdas for composite preconditioners
    auto makeBJ = [&]() -> std::shared_ptr<const gko::LinOp>
    {
        return gko::preconditioner::Jacobi<double, int>::build()
            .with_max_block_size(static_cast<unsigned>(blockSize_))
            .on(exec)
            ->generate(operatorF64_->localMatrix());
    };

    auto makeISAI = [&]() -> std::shared_ptr<const gko::LinOp>
    {
        return gko::preconditioner::SpdIsai<double, int>::build()
            .with_sparsity_power(isaiSparsityPower_)
            .on(exec)
            ->generate(operatorF64_->localMatrix());
    };

    // Create preconditioner and solver
    std::unique_ptr<gko::LinOp> solver;

    if (preconditionerType_ == PreconditionerType::BLOCK_JACOBI_ISAI)
    {
        auto bj = makeBJ();
        auto isai = makeISAI();
        std::vector<std::shared_ptr<const gko::LinOp>> ops = {isai, bj};
        auto composed = gko::share(
            gko::Composition<double>::create(ops.begin(), ops.end())
        );
        solver = gko::solver::Cg<double>::build()
            .with_generated_preconditioner(composed)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF64_->localMatrix());
    }
    else if (preconditionerType_ == PreconditionerType::BJ_ISAI_SANDWICH)
    {
        auto bj = makeBJ();
        auto isai = makeISAI();
        std::vector<std::shared_ptr<const gko::LinOp>> ops = {bj, isai, bj};
        auto composed = gko::share(
            gko::Composition<double>::create(ops.begin(), ops.end())
        );
        solver = gko::solver::Cg<double>::build()
            .with_generated_preconditioner(composed)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF64_->localMatrix());
    }
    else if (preconditionerType_ == PreconditionerType::BJ_ISAI_ADDITIVE)
    {
        auto bj = makeBJ();
        auto isai = makeISAI();
        auto additive = gko::share(
            AdditiveLinOp<double>::create(bj, isai)
        );
        solver = gko::solver::Cg<double>::build()
            .with_generated_preconditioner(additive)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF64_->localMatrix());
    }
    else if (preconditionerType_ == PreconditionerType::BJ_ISAI_GMRES)
    {
        auto bj = makeBJ();
        auto isai = makeISAI();
        std::vector<std::shared_ptr<const gko::LinOp>> ops = {isai, bj};
        auto composed = gko::share(
            gko::Composition<double>::create(ops.begin(), ops.end())
        );
        solver = gko::solver::Gmres<double>::build()
            .with_krylov_dim(100u)
            .with_generated_preconditioner(composed)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF64_->localMatrix());
    }
    else if (preconditionerType_ == PreconditionerType::BJ_ISAI_INNER_OUTER)
    {
        auto bjFactory = gko::share(
            gko::preconditioner::Jacobi<double, int>::build()
                .with_max_block_size(static_cast<unsigned>(blockSize_))
                .on(exec)
        );
        auto innerCrit = gko::share(
            gko::stop::Iteration::build()
                .with_max_iters(static_cast<gko::size_type>(5))
                .on(exec)
        );
        auto innerSolver = gko::share(
            gko::solver::Cg<double>::build()
                .with_preconditioner(bjFactory)
                .with_criteria(innerCrit)
                .on(exec)
        );
        solver = gko::solver::Gmres<double>::build()
            .with_krylov_dim(100u)
            .with_flexible(true)
            .with_preconditioner(innerSolver)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF64_->localMatrix());
    }
    else if (preconditionerType_ == PreconditionerType::FFT)
    {
        auto nCells = static_cast<gko::size_type>(
            fftDimensions_.x() * fftDimensions_.y() * fftDimensions_.z()
        );
        auto fftPrecond = gko::share(
            FFTPreconditioner<double>::create(
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

        solver = gko::solver::Cg<double>::build()
            .with_generated_preconditioner(fftPrecond)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF64_->localMatrix());
    }
    else if (preconditionerType_ == PreconditionerType::FFT_BLOCK_JACOBI)
    {
        auto nCells = static_cast<gko::size_type>(
            fftDimensions_.x() * fftDimensions_.y() * fftDimensions_.z()
        );
        auto fft = gko::share(
            FFTPreconditioner<double>::create(
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
        auto bj = makeBJ();
        auto additive = gko::share(
            AdditiveLinOp<double>::create(fft, bj)
        );

        solver = gko::solver::Cg<double>::build()
            .with_generated_preconditioner(additive)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF64_->localMatrix());
    }
    else
    {
        std::shared_ptr<gko::LinOpFactory> precond;
        switch (preconditionerType_)
        {
            case PreconditionerType::BLOCK_JACOBI:
                precond = gko::share(
                    gko::preconditioner::Jacobi<double, int>::build()
                        .with_max_block_size(
                            static_cast<unsigned>(blockSize_)
                        )
                        .on(exec)
                );
                break;
            case PreconditionerType::ISAI:
                precond = gko::share(
                    gko::preconditioner::SpdIsai<double, int>::build()
                        .with_sparsity_power(isaiSparsityPower_)
                        .on(exec)
                );
                break;
            case PreconditionerType::JACOBI:
            default:
                precond = gko::share(
                    gko::preconditioner::Jacobi<double, int>::build()
                        .with_max_block_size(1u)
                        .on(exec)
                );
                break;
        }
        solver = gko::solver::Cg<double>::build()
            .with_preconditioner(precond)
            .with_criteria(iterCrit, resCrit)
            .on(exec)
            ->generate(operatorF64_->localMatrix());
    }

    auto logger = gko::share(gko::log::Convergence<double>::create());
    solver->add_logger(logger);

    solver->apply(b.get(), x.get());

    label numIters = static_cast<label>(logger->get_num_iterations());

    // Copy solution back
    FP32CastWrapper::fromGinkgoF64(x.get(), psi);

    return numIters;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::OGL::OGLBatchedSolver::OGLBatchedSolver
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
    batchSize_(3),  // Default for vector fields
    operatorF32_(nullptr),
    operatorF64_(nullptr),
    currentBatchIndex_(0),
    batchedModeActive_(false)
{
    // Read batch size from controls
    if (solverControls.found("OGLCoeffs"))
    {
        const dictionary& oglCoeffs = solverControls.subDict("OGLCoeffs");
        batchSize_ = oglCoeffs.lookupOrDefault<label>("batchSize", 3);
    }

    if (debug_ >= 1)
    {
        Info<< "OGLBatchedSolver: Created for field " << fieldName << nl
            << "  batchSize: " << batchSize_ << nl
            << "  preconditioner: "
            << [&]() -> const char* {
                switch (preconditionerType_)
                {
                    case PreconditionerType::JACOBI: return "Jacobi";
                    case PreconditionerType::BLOCK_JACOBI: return "blockJacobi";
                    case PreconditionerType::ISAI: return "ISAI";
                    case PreconditionerType::BLOCK_JACOBI_ISAI: return "blockJacobiISAI";
                    case PreconditionerType::BJ_ISAI_SANDWICH: return "bjIsaiSandwich";
                    case PreconditionerType::BJ_ISAI_ADDITIVE: return "bjIsaiAdditive";
                    case PreconditionerType::BJ_ISAI_GMRES: return "bjIsaiGmres";
                    case PreconditionerType::BJ_ISAI_INNER_OUTER: return "bjIsaiInnerOuter";
                    case PreconditionerType::FFT: return "FFT";
                    case PreconditionerType::FFT_BLOCK_JACOBI: return "fftBlockJacobi";
                    default: return "unknown";
                }
            }()
            << nl
            << endl;
    }
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::OGL::OGLBatchedSolver::beginBatch()
{
    currentBatchIndex_ = 0;
    batchedModeActive_ = true;
}


void Foam::OGL::OGLBatchedSolver::addToBatch
(
    scalarField& psi,
    const scalarField& source
)
{
    if (!batchedModeActive_)
    {
        FatalErrorInFunction
            << "Batch mode not active. Call beginBatch() first."
            << abort(FatalError);
    }

    if (currentBatchIndex_ >= batchSize_)
    {
        FatalErrorInFunction
            << "Batch overflow. Batch size is " << batchSize_
            << abort(FatalError);
    }

    // Initialize storage on first add
    if (currentBatchIndex_ == 0)
    {
        initBatch(psi.size());
    }

    // Copy data to batch storage
    batchedRHS_[currentBatchIndex_] = source;
    batchedSolution_[currentBatchIndex_] = psi;

    currentBatchIndex_++;
}


Foam::label Foam::OGL::OGLBatchedSolver::endBatch()
{
    if (!batchedModeActive_)
    {
        return 0;
    }

    label totalIters = executeBatchedSolve();

    batchedModeActive_ = false;
    currentBatchIndex_ = 0;

    return totalIters;
}


// ************************************************************************* //
