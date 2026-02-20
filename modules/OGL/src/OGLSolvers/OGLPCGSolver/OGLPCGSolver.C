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
#include "AdditiveLinOp.H"
#include "FFTPreconditioner.H"
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
    defineTypeNameAndDebug(OGLPCGSolver, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<OGLPCGSolver>
        addOGLPCGSymMatrixConstructorToTable_;
}
}

// Static cache members
std::map<Foam::word, Foam::OGL::OGLPCGSolver::SolverCache>
    Foam::OGL::OGLPCGSolver::cache_;
std::mutex Foam::OGL::OGLPCGSolver::cacheMutex_;
std::map<Foam::word, Foam::label> Foam::OGL::OGLPCGSolver::mgCallCount_;
std::map<Foam::word, Foam::label> Foam::OGL::OGLPCGSolver::mgLastIters_;


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<typename ValueType>
void Foam::OGL::OGLPCGSolver::updateSolverImpl
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
                Info<< "OGLPCGSolver: Reusing cached operator" << endl;
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

        // Create preconditioner and solver based on configuration.
        // The composite case (BLOCK_JACOBI_ISAI) uses
        // with_generated_preconditioner() since we need to build both
        // preconditioners from the matrix first and compose them.
        // Helper lambdas: build BJ and ISAI from the CSR matrix
        auto makeBJ = [&]() -> std::shared_ptr<const gko::LinOp>
        {
            return gko::preconditioner::Jacobi<ValueType, int>::build()
                .with_max_block_size(static_cast<unsigned>(blockSize_))
                .on(exec)
                ->generate(op->localMatrix());
        };

        auto makeISAI = [&]() -> std::shared_ptr<const gko::LinOp>
        {
            return gko::preconditioner::SpdIsai<ValueType, int>::build()
                .with_sparsity_power(isaiSparsityPower_)
                .on(exec)
                ->generate(op->localMatrix());
        };

        if (preconditionerType_ == PreconditionerType::BLOCK_JACOBI_ISAI)
        {
            // Multiplicative: z = ISAI(BJ(r)) — CG
            auto bj = makeBJ();
            auto isai = makeISAI();
            std::vector<std::shared_ptr<const gko::LinOp>> ops =
                {isai, bj};
            auto composed = gko::share(
                gko::Composition<ValueType>::create(
                    ops.begin(), ops.end()
                )
            );

            solver = gko::solver::Cg<ValueType>::build()
                .with_generated_preconditioner(composed)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());
        }
        else if
        (
            preconditionerType_ == PreconditionerType::BJ_ISAI_SANDWICH
        )
        {
            // Symmetric: z = BJ(ISAI(BJ(r))) — preserves SPD for CG
            auto bj = makeBJ();
            auto isai = makeISAI();
            std::vector<std::shared_ptr<const gko::LinOp>> ops =
                {bj, isai, bj};
            auto composed = gko::share(
                gko::Composition<ValueType>::create(
                    ops.begin(), ops.end()
                )
            );

            solver = gko::solver::Cg<ValueType>::build()
                .with_generated_preconditioner(composed)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());
        }
        else if
        (
            preconditionerType_ == PreconditionerType::BJ_ISAI_ADDITIVE
        )
        {
            // Additive: z = BJ(r) + ISAI(r) — SPD preserved
            auto bj = makeBJ();
            auto isai = makeISAI();
            auto additive = gko::share(
                AdditiveLinOp<ValueType>::create(bj, isai)
            );

            solver = gko::solver::Cg<ValueType>::build()
                .with_generated_preconditioner(additive)
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
        else if
        (
            preconditionerType_ == PreconditionerType::BJ_ISAI_INNER_OUTER
        )
        {
            // Inner CG-BJ as preconditioner for outer FGMRES
            auto bjFactory = gko::share(
                gko::preconditioner::Jacobi<ValueType, int>::build()
                    .with_max_block_size(
                        static_cast<unsigned>(blockSize_)
                    )
                    .on(exec)
            );

            auto innerCrit = gko::share(
                gko::stop::Iteration::build()
                    .with_max_iters(static_cast<gko::size_type>(5))
                    .on(exec)
            );

            auto innerSolver = gko::share(
                gko::solver::Cg<ValueType>::build()
                    .with_preconditioner(bjFactory)
                    .with_criteria(innerCrit)
                    .on(exec)
            );

            solver = gko::solver::Gmres<ValueType>::build()
                .with_krylov_dim(100u)
                .with_flexible(true)
                .with_preconditioner(innerSolver)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());
        }
        else if (preconditionerType_ == PreconditionerType::FFT)
        {
            // FFT Laplacian preconditioner (structured Cartesian grids)
            auto nCells = static_cast<gko::size_type>(
                fftDimensions_.x() * fftDimensions_.y() * fftDimensions_.z()
            );
            auto fftPrecond = gko::share(
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

            // Extract mean coupling coefficients from CSR matrix
            // to update FFT eigenvalues with actual matrix values
            {
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

                // For structured mesh: x-neighbors differ by ±1,
                // y-neighbors by ±nx, z-neighbors by ±(nx*ny)
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

                Info<< "FFT: extracted mean couplings from CSR:"
                    << " coeffX=" << meanCoeffX
                    << " (n=" << countX << ")"
                    << " coeffY=" << meanCoeffY
                    << " (n=" << countY << ")"
                    << " coeffZ=" << meanCoeffZ
                    << " (n=" << countZ << ")"
                    << endl;

                // Update FFT preconditioner eigenvalues with actual
                // coefficients
                fftPrecond->updateCoeffs(
                    meanCoeffX, meanCoeffY, meanCoeffZ
                );
            }

            solver = gko::solver::Cg<ValueType>::build()
                .with_generated_preconditioner(fftPrecond)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());
        }
        else if
        (
            preconditionerType_ == PreconditionerType::FFT_BLOCK_JACOBI
        )
        {
            // FFT + Block Jacobi additive (structured Cartesian grids)
            auto nCells = static_cast<gko::size_type>(
                fftDimensions_.x() * fftDimensions_.y() * fftDimensions_.z()
            );
            auto fft = gko::share(
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
            auto bj = makeBJ();
            auto additive = gko::share(
                AdditiveLinOp<ValueType>::create(fft, bj)
            );

            solver = gko::solver::Cg<ValueType>::build()
                .with_generated_preconditioner(additive)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());
        }
        else if (preconditionerType_ == PreconditionerType::MULTIGRID)
        {
            // AMG preconditioner: 1 V-cycle of Multigrid as PCG preconditioner.
            // Coarsening: Parallel Graph Matching (PGM).
            // Smoother: Chebyshev (no inner products) or Jacobi.

            // Build smoother factory
            std::shared_ptr<gko::LinOpFactory> smootherFactory;

            if (mgSmoother_ == "chebyshev")
            {
                // Jacobi-preconditioned Chebyshev polynomial smoother.
                //
                // For SPD M-matrices (discrete Laplacian in any dimension),
                // λ_max(D⁻¹A) ≤ 2 universally (Gershgorin: row sums of
                // D⁻¹A = 1 + off-diag/diag ≤ 2 by diagonal dominance).
                // Galerkin coarse-grid operators preserve the M-matrix
                // property, so these fixed foci work at ALL multigrid levels.
                //
                // Standard AMG Chebyshev (cf. hypre, PETSc):
                //   upper = 1.1 * λ_max(D⁻¹A) = 2.2  (10% safety margin)
                //   lower = upper / 30 ≈ 0.073         (standard ratio)
                constexpr ValueType chebyUpper = static_cast<ValueType>(2.2);
                constexpr ValueType chebyLower =
                    chebyUpper / static_cast<ValueType>(30);

                auto jacInner = gko::share(
                    gko::preconditioner::Jacobi<ValueType, int>::build()
                        .with_max_block_size(1u)
                        .on(exec)
                );

                auto chebyFactory = gko::share(
                    gko::solver::Chebyshev<ValueType>::build()
                        .with_foci(std::pair<ValueType, ValueType>(
                            chebyLower, chebyUpper
                        ))
                        .with_preconditioner(jacInner)
                        .with_criteria(
                            gko::stop::Iteration::build()
                                .with_max_iters(
                                    static_cast<gko::size_type>(
                                        mgSmootherIters_
                                    )
                                )
                                .on(exec)
                        )
                        .on(exec)
                );
                smootherFactory = chebyFactory;
            }
            else if (mgSmoother_ == "blockJacobi")
            {
                auto bjFactory = gko::share(
                    gko::preconditioner::Jacobi<ValueType, int>::build()
                        .with_max_block_size(
                            static_cast<unsigned>(blockSize_)
                        )
                        .on(exec)
                );
                smootherFactory = gko::share(
                    gko::solver::build_smoother(
                        bjFactory,
                        static_cast<gko::size_type>(mgSmootherIters_),
                        static_cast<ValueType>(mgSmootherRelax_)
                    )
                );
            }
            else
            {
                // Default: scalar Jacobi smoother
                auto jacFactory = gko::share(
                    gko::preconditioner::Jacobi<ValueType, int>::build()
                        .with_max_block_size(1u)
                        .on(exec)
                );
                smootherFactory = gko::share(
                    gko::solver::build_smoother(
                        jacFactory,
                        static_cast<gko::size_type>(mgSmootherIters_),
                        static_cast<ValueType>(mgSmootherRelax_)
                    )
                );
            }

            // PGM coarsening (deterministic for reproducibility)
            auto pgmFactory = gko::share(
                gko::multigrid::Pgm<ValueType, int>::build()
                    .with_deterministic(true)
                    .on(exec)
            );

            // 1 V-cycle multigrid as preconditioner
            auto mgFactory = gko::share(
                gko::solver::Multigrid::build()
                    .with_mg_level(pgmFactory)
                    .with_pre_smoother(smootherFactory)
                    .with_post_uses_pre(true)
                    .with_max_levels(
                        static_cast<gko::size_type>(mgMaxLevels_)
                    )
                    .with_min_coarse_rows(
                        static_cast<gko::size_type>(mgMinCoarseRows_)
                    )
                    .with_criteria(
                        gko::stop::Iteration::build()
                            .with_max_iters(1u)
                            .on(exec)
                    )
                    .on(exec)
            );

            // Time the hierarchy generation (PGM coarsening + Galerkin ops)
            exec->synchronize();

            // VRAM profiling: measure before generate
            size_t vramFreeBefore = 0, vramTotal = 0;
            haloGetGpuMemInfo(&vramFreeBefore, &vramTotal);

            auto mgStart = std::chrono::high_resolution_clock::now();

            solver = gko::solver::Cg<ValueType>::build()
                .with_preconditioner(mgFactory)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());

            exec->synchronize();
            auto mgEnd = std::chrono::high_resolution_clock::now();

            // VRAM profiling: measure after generate
            size_t vramFreeAfter = 0, vramDummy = 0;
            haloGetGpuMemInfo(&vramFreeAfter, &vramDummy);
            // Use signed arithmetic to handle freed memory (old cache released)
            long mgVramDelta =
                static_cast<long>(vramFreeBefore)
              - static_cast<long>(vramFreeAfter);

            double mgSetupMs = std::chrono::duration<double, std::milli>(
                mgEnd - mgStart
            ).count();

            if (debug_ >= 1)
            {
                Info<< "OGLPCGSolver: MG hierarchy setup: "
                    << mgSetupMs << " ms, VRAM delta: "
                    << mgVramDelta / (1024*1024) << " MB (total: "
                    << vramTotal / (1024*1024) << " MB, free: "
                    << vramFreeAfter / (1024*1024) << " MB)" << endl;
            }
        }
        else
        {
            // Factory-based preconditioner path
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

                case PreconditionerType::ISAI:
                    precond = gko::share(
                        gko::preconditioner::SpdIsai<ValueType, int>::build()
                            .with_sparsity_power(isaiSparsityPower_)
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

            // Generate solver from the local CSR matrix.  Preconditioners
            // (Jacobi, Block Jacobi, ISAI) need matrix data which CSR
            // supports but the matrix-free FoamGinkgoLinOp does not.
            // The outer iterative refinement loop handles interface
            // contributions in FP64.
            solver = gko::solver::Cg<ValueType>::build()
                .with_preconditioner(precond)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());
        }
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
    std::shared_ptr<gko::LinOp>& solver,
    ValueType tolerance
) const
{
    auto exec = OGLExecutor::instance().executor();
    label numIters = 0;

    try
    {
        // Decide whether to rebuild the solver or reuse the cached one.
        //
        // For multigrid with mgCacheInterval > 0, we can reuse the
        // solver (including its stale MG hierarchy) between rebuilds.
        // CG correctness is maintained because CG only requires the
        // system matrix (A*x) to be correct — the preconditioner
        // affects convergence rate, not the solution.  The stale MG
        // hierarchy is still SPD and a reasonable preconditioner as
        // long as the matrix hasn't changed dramatically.
        //
        // Rebuild is triggered by:
        //  (a) no cached solver exists (first call), or
        //  (b) call counter reaches mgCacheInterval, or
        //  (c) previous solve exceeded mgCacheMaxIters
        bool needRebuild = true;

        if
        (
            solver
         && preconditionerType_ == PreconditionerType::MULTIGRID
         && mgCacheInterval_ > 0
        )
        {
            // Check if we can reuse cached solver
            label callCount = 0;
            label lastIters = 0;
            {
                std::lock_guard<std::mutex> lock(cacheMutex_);
                callCount = mgCallCount_[fieldName_];
                lastIters = mgLastIters_[fieldName_];
            }

            // Warm-up: always rebuild during the first interval to
            // establish a stable matrix pattern before caching.
            // This avoids catastrophic divergence during startup
            // transients where the matrix changes dramatically.
            bool warmupDone = callCount >= mgCacheInterval_;
            bool intervalOk = (callCount % mgCacheInterval_) != 0;
            bool itersOk = lastIters < mgCacheMaxIters_;

            if (warmupDone && intervalOk && itersOk)
            {
                // Reuse cached solver — just update CSR values
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

                // Force CSR value refresh for the system matrix.
                // The cached CG solver references this CSR object,
                // so updating values in-place is sufficient.
                op->localMatrix();

                needRebuild = false;

                if (debug_ >= 1)
                {
                    Info<< "OGLPCGSolver: Reusing cached MG hierarchy"
                        << " (call " << callCount
                        << ", last iters " << lastIters << ")"
                        << endl;
                }
            }
            else if (debug_ >= 1)
            {
                Info<< "OGLPCGSolver: Rebuilding MG hierarchy"
                    << " (call " << callCount
                    << ", interval " << mgCacheInterval_
                    << ", last iters " << lastIters
                    << ", max " << mgCacheMaxIters_ << ")"
                    << endl;
            }
        }

        if (needRebuild)
        {
            if (op)
            {
                // Operator exists — update pointers and invalidate
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

            // Full rebuild: operator + solver + preconditioner
            updateSolverImpl(op, solver, tolerance);
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
        // gko::share converts unique_ptr to shared_ptr for add_logger
        auto logger = gko::share(gko::log::Convergence<ValueType>::create());
        solver->add_logger(logger);

        // Solve with error handling — time the CG iterations
        exec->synchronize();
        auto solveStart = std::chrono::high_resolution_clock::now();

        solver->apply(b.get(), x.get());

        // Synchronize to ensure solve completed
        OGLExecutor::instance().synchronize();

        auto solveEnd = std::chrono::high_resolution_clock::now();
        double solveMs = std::chrono::duration<double, std::milli>(
            solveEnd - solveStart
        ).count();

        // Get iteration count
        numIters = static_cast<label>(logger->get_num_iterations());

        // Update MG call counter and iteration tracking
        if (preconditionerType_ == PreconditionerType::MULTIGRID)
        {
            std::lock_guard<std::mutex> lock(cacheMutex_);
            mgCallCount_[fieldName_]++;
            mgLastIters_[fieldName_] = numIters;
        }

        if (debug_ >= 1)
        {
            Info<< "OGLPCGSolver: CG solve: " << solveMs << " ms, "
                << numIters << " iters, "
                << (numIters > 0 ? solveMs / numIters : 0)
                << " ms/iter" << endl;
        }

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
    // Fallback path: when called outside the main solve() flow,
    // use relTol if available, otherwise tolerance_ as reduction factor.
    // The main solve path passes the correct effective factor via solveFP64().
    const double ginkgoTol = (relTol_ > 0) ? relTol_ : tolerance_;
    updateSolverImpl<double>(operatorF64_, solverF64_, ginkgoTol);
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

Foam::label Foam::OGL::OGLPCGSolver::solveFP32
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

    // Persist to static cache.  For multigrid with caching enabled,
    // also persist the solver (including MG hierarchy) for reuse.
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[fieldName_].opF32 = operatorF32_;

        if
        (
            preconditionerType_ == PreconditionerType::MULTIGRID
         && mgCacheInterval_ > 0
        )
        {
            cache_[fieldName_].solverF32 = solverF32_;
        }
    }

    return iters;
}


Foam::label Foam::OGL::OGLPCGSolver::solveFP64
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

    // Persist to static cache.  For multigrid with caching enabled,
    // also persist the solver (including MG hierarchy) for reuse.
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[fieldName_].opF64 = operatorF64_;

        if
        (
            preconditionerType_ == PreconditionerType::MULTIGRID
         && mgCacheInterval_ > 0
        )
        {
            cache_[fieldName_].solverF64 = solverF64_;
        }
    }

    return iters;
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
    // Restore cached state from previous instantiation (if any).
    // Always restore operators (GPU CSR structure).
    // For multigrid with caching: also restore solver (MG hierarchy).
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto it = cache_.find(fieldName);
        if (it != cache_.end())
        {
            operatorF32_ = it->second.opF32;
            operatorF64_ = it->second.opF64;

            if
            (
                preconditionerType_ == PreconditionerType::MULTIGRID
             && mgCacheInterval_ > 0
            )
            {
                solverF32_ = it->second.solverF32;
                solverF64_ = it->second.solverF64;
            }
            // else solverF32_/solverF64_ left as nullptr — rebuilt each call

            if (debug_ >= 2)
            {
                Info<< "OGLPCGSolver: Restored cached "
                    << (solverF64_ ? "solver+operator" : "operator")
                    << " for " << fieldName << endl;
            }
        }
    }

    if (debug_ >= 1)
    {
        Info<< "OGLPCGSolver: Created for field " << fieldName << nl
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
                    case PreconditionerType::ISAI: return "ISAI";
                    case PreconditionerType::BLOCK_JACOBI_ISAI: return "blockJacobiISAI";
                    case PreconditionerType::BJ_ISAI_SANDWICH: return "bjIsaiSandwich";
                    case PreconditionerType::BJ_ISAI_ADDITIVE: return "bjIsaiAdditive";
                    case PreconditionerType::BJ_ISAI_GMRES: return "bjIsaiGmres";
                    case PreconditionerType::BJ_ISAI_INNER_OUTER: return "bjIsaiInnerOuter";
                    case PreconditionerType::FFT: return "FFT";
                    case PreconditionerType::FFT_BLOCK_JACOBI: return "fftBlockJacobi";
                    case PreconditionerType::MULTIGRID: return "multigrid";
                    default: return "unknown";
                }
            }()
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
