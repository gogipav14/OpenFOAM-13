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
#include "GeometricMultigridLinOp.H"
#include "HaloKernels.h"
#include "addToRunTimeSelectionTable.H"

#include <chrono>
#include <cmath>
#include <limits>

// * * * * * * * * * * * * Custom Chebyshev Polynomial LinOp * * * * * * * * //

namespace Foam
{
namespace OGL
{

// Chebyshev polynomial smoother as a Ginkgo LinOp.
// Applies k steps of preconditioned Richardson iteration with Chebyshev
// root-based relaxation parameters for optimal polynomial smoothing.
//
// Given eigenvalue bounds [a, b] of M^{-1}A, the k relaxation parameters
// are omega_j = 1 / (d - c * cos(pi*(2j-1)/(2k)))
// where d = (a+b)/2, c = (b-a)/2.
//
// Each apply() computes: z = p_k(M^{-1}A) * M^{-1} * r
// starting from x=0. No inner products — works with NSD matrices.
template <typename ValueType>
class ChebyshevPolyLinOp
:
    public gko::EnableLinOp<ChebyshevPolyLinOp<ValueType>>,
    public gko::EnableCreateMethod<ChebyshevPolyLinOp<ValueType>>
{
    friend class gko::EnablePolymorphicObject<ChebyshevPolyLinOp, gko::LinOp>;
    friend class gko::EnableCreateMethod<ChebyshevPolyLinOp>;

public:

    using value_type = ValueType;

protected:

    explicit ChebyshevPolyLinOp(std::shared_ptr<const gko::Executor> exec)
    :
        gko::EnableLinOp<ChebyshevPolyLinOp>(exec),
        degree_(1)
    {}

    ChebyshevPolyLinOp
    (
        std::shared_ptr<const gko::Executor> exec,
        gko::dim<2> size,
        std::shared_ptr<const gko::LinOp> systemMatrix,
        std::shared_ptr<const gko::LinOp> innerPrecond,
        int degree,
        ValueType eigMin,
        ValueType eigMax
    )
    :
        gko::EnableLinOp<ChebyshevPolyLinOp>(exec, size),
        system_(systemMatrix),
        precond_(innerPrecond),
        degree_(degree)
    {
        // Precompute Chebyshev root-based relaxation parameters
        ValueType d = (eigMax + eigMin) / ValueType(2);
        ValueType c = (eigMax - eigMin) / ValueType(2);
        omegas_.resize(degree);
        for (int j = 0; j < degree; j++)
        {
            ValueType cosArg = std::cos(
                M_PI * ValueType(2*j + 1) / ValueType(2*degree)
            );
            omegas_[j] = ValueType(1) / (d - c * cosArg);
        }
    }

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        using Dense = gko::matrix::Dense<ValueType>;
        auto exec = this->get_executor();
        auto dense_b = gko::as<Dense>(b);
        auto dense_x = gko::as<Dense>(x);
        auto n = dense_b->get_size()[0];

        auto one = gko::initialize<Dense>({ValueType(1)}, exec);
        auto neg_one = gko::initialize<Dense>({ValueType(-1)}, exec);

        // Work vectors
        auto r = Dense::create(exec, gko::dim<2>{n, 1});
        auto z = Dense::create(exec, gko::dim<2>{n, 1});
        auto normVec = Dense::create(
            exec->get_master(), gko::dim<2>{1, 1}
        );

        // x = 0
        dense_x->fill(ValueType(0));

        for (int j = 0; j < degree_; j++)
        {
            // r = b - A*x
            r->copy_from(dense_b);
            system_->apply(neg_one.get(), dense_x, one.get(), r.get());

            // z = M^{-1} * r
            precond_->apply(r.get(), z.get());

            // x += omega_j * z
            auto omegaVec = gko::initialize<Dense>(
                {omegas_[j]}, exec
            );
            dense_x->add_scaled(omegaVec.get(), z.get());

            // One-shot diagnostic
            if (callCount_ == 0)
            {
                r->compute_norm2(normVec.get());
                double rNorm = static_cast<double>(
                    normVec->get_const_values()[0]
                );
                dense_x->compute_norm2(normVec.get());
                double xNorm = static_cast<double>(
                    normVec->get_const_values()[0]
                );
                fprintf(stderr,
                    "ChebyPoly step %d/%d: omega=%.4f"
                    " ||r||=%.6e ||x||=%.6e\n",
                    j, degree_,
                    static_cast<double>(omegas_[j]),
                    rNorm, xNorm);
            }
        }
        callCount_++;
    }

    void apply_impl
    (
        const gko::LinOp* alpha,
        const gko::LinOp* b,
        const gko::LinOp* beta,
        gko::LinOp* x
    ) const override
    {
        using Dense = gko::matrix::Dense<ValueType>;
        auto dense_x = gko::as<Dense>(x);
        auto temp = Dense::create(
            this->get_executor(), dense_x->get_size()
        );
        this->apply_impl(b, temp.get());
        auto dense_alpha = gko::as<Dense>(alpha);
        auto dense_beta = gko::as<Dense>(beta);
        dense_x->scale(dense_beta);
        dense_x->add_scaled(dense_alpha, temp);
    }

private:

    std::shared_ptr<const gko::LinOp> system_;
    std::shared_ptr<const gko::LinOp> precond_;
    int degree_;
    std::vector<ValueType> omegas_;
    mutable int callCount_ = 0;
};


// * * * * * * * * * * * * * 2-Level V-Cycle LinOp  * * * * * * * * * * * * //

// Symmetric 2-level V-cycle preconditioner as a Ginkgo LinOp.
// V-cycle: pre-smooth(BJ) -> coarse-correct(FFT) -> post-smooth(BJ).
//
// Symmetric by construction when smoother, coarse solver, and system
// matrix are all symmetric. This makes it compatible with CG for NSD
// pressure systems.
//
// The V-cycle combines the strengths of both preconditioners:
//   - BJ handles high-frequency/local modes (boundaries, variable coeffs)
//   - FFT handles low-frequency/global modes (bulk Laplacian structure)
//
// Expected: ~5-10 CG iterations (vs 19-21 with additive FFT+BJ).
template <typename ValueType>
class VCycleLinOp
:
    public gko::EnableLinOp<VCycleLinOp<ValueType>>,
    public gko::EnableCreateMethod<VCycleLinOp<ValueType>>
{
    friend class gko::EnablePolymorphicObject<VCycleLinOp, gko::LinOp>;
    friend class gko::EnableCreateMethod<VCycleLinOp>;

public:

    using value_type = ValueType;

protected:

    explicit VCycleLinOp(std::shared_ptr<const gko::Executor> exec)
    :
        gko::EnableLinOp<VCycleLinOp>(exec)
    {}

    VCycleLinOp
    (
        std::shared_ptr<const gko::Executor> exec,
        gko::dim<2> size,
        std::shared_ptr<const gko::LinOp> systemMatrix,
        std::shared_ptr<const gko::LinOp> smoother,
        std::shared_ptr<const gko::LinOp> coarseSolver,
        int preSmooth = 1,
        int postSmooth = 1
    )
    :
        gko::EnableLinOp<VCycleLinOp>(exec, size),
        system_(systemMatrix),
        smoother_(smoother),
        coarse_(coarseSolver),
        preSmooth_(preSmooth),
        postSmooth_(postSmooth)
    {
        auto n = size[0];
        // Pre-allocate work vectors (reused every apply call)
        r_ = gko::matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{n, 1}
        );
        z_ = gko::matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{n, 1}
        );
        one_ = gko::initialize<gko::matrix::Dense<ValueType>>(
            {ValueType(1)}, exec
        );
        neg_one_ = gko::initialize<gko::matrix::Dense<ValueType>>(
            {ValueType(-1)}, exec
        );
    }

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        using Dense = gko::matrix::Dense<ValueType>;
        auto dense_b = gko::as<Dense>(b);
        auto dense_x = gko::as<Dense>(x);

        // === Pre-smooth: repeated BJ Richardson iterations ===
        // First iteration: x = BJ^{-1} * b
        smoother_->apply(dense_b, dense_x);
        // Subsequent iterations: x += BJ^{-1} * (b - A*x)
        for (int i = 1; i < preSmooth_; i++)
        {
            r_->copy_from(dense_b);
            system_->apply(neg_one_.get(), dense_x, one_.get(), r_.get());
            smoother_->apply(r_.get(), z_.get());
            dense_x->add_scaled(one_.get(), z_.get());
        }

        // === Residual after pre-smooth: r = b - A*x ===
        r_->copy_from(dense_b);
        system_->apply(neg_one_.get(), dense_x, one_.get(), r_.get());

        // === Coarse correct: z = FFT^{-1} * r ===
        coarse_->apply(r_.get(), z_.get());
        dense_x->add_scaled(one_.get(), z_.get());

        // === Post-smooth: repeated BJ Richardson iterations ===
        for (int i = 0; i < postSmooth_; i++)
        {
            r_->copy_from(dense_b);
            system_->apply(neg_one_.get(), dense_x, one_.get(), r_.get());
            smoother_->apply(r_.get(), z_.get());
            dense_x->add_scaled(one_.get(), z_.get());
        }

        // One-shot diagnostic
        if (callCount_ == 0)
        {
            auto normVec = Dense::create(
                this->get_executor()->get_master(), gko::dim<2>{1, 1}
            );
            dense_x->compute_norm2(normVec.get());
            double xNorm = static_cast<double>(
                normVec->get_const_values()[0]
            );
            dense_b->compute_norm2(normVec.get());
            double bNorm = static_cast<double>(
                normVec->get_const_values()[0]
            );
            // Final residual after full V-cycle
            r_->copy_from(dense_b);
            system_->apply(neg_one_.get(), dense_x, one_.get(), r_.get());
            r_->compute_norm2(normVec.get());
            double rNorm = static_cast<double>(
                normVec->get_const_values()[0]
            );
            fprintf(stderr,
                "VCycle: ||b||=%.4e ||x||=%.4e ||b-Ax||=%.4e"
                " reduction=%.4f\n",
                bNorm, xNorm, rNorm,
                (bNorm > 1e-30) ? rNorm / bNorm : 0.0);
        }
        callCount_++;
    }

    void apply_impl
    (
        const gko::LinOp* alpha,
        const gko::LinOp* b,
        const gko::LinOp* beta,
        gko::LinOp* x
    ) const override
    {
        using Dense = gko::matrix::Dense<ValueType>;
        auto dense_x = gko::as<Dense>(x);
        auto temp = Dense::create(
            this->get_executor(), dense_x->get_size()
        );
        this->apply_impl(b, temp.get());
        auto dense_alpha = gko::as<Dense>(alpha);
        auto dense_beta = gko::as<Dense>(beta);
        dense_x->scale(dense_beta);
        dense_x->add_scaled(dense_alpha, temp);
    }

private:

    std::shared_ptr<const gko::LinOp> system_;
    std::shared_ptr<const gko::LinOp> smoother_;
    std::shared_ptr<const gko::LinOp> coarse_;
    int preSmooth_;
    int postSmooth_;

    // Pre-allocated work vectors (shared_ptr for Ginkgo copy-assign compat)
    mutable std::shared_ptr<gko::matrix::Dense<ValueType>> r_;
    mutable std::shared_ptr<gko::matrix::Dense<ValueType>> z_;
    mutable std::shared_ptr<gko::matrix::Dense<ValueType>> one_;
    mutable std::shared_ptr<gko::matrix::Dense<ValueType>> neg_one_;
    mutable int callCount_ = 0;
};


} // End namespace OGL
} // End namespace Foam

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
        else if (preconditionerType_ == PreconditionerType::FFT
              || preconditionerType_ == PreconditionerType::FFT_DIRECT
              || preconditionerType_ == PreconditionerType::FFT_CHEBYSHEV
              || preconditionerType_ == PreconditionerType::FFT_SCALED)
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

                // --- rAU spectral analysis ---
                // Extract per-face x-coupling coefficient (rAU * dy*dz/dx)
                // and compute its Fourier spectrum to understand which
                // modes the FFT preconditioner misses.
                if (debug_ >= 1)
                {
                    int ny = fftDimensions_.y();
                    int nz = fftDimensions_.z();
                    int nxyz = nx * ny * nz;

                    // Extract per-face x-coupling for each cell
                    // (average of left and right x-face values)
                    std::vector<double> coeffXPerCell(nxyz, 0.0);
                    std::vector<double> coeffYPerCell(nxyz, 0.0);
                    std::vector<double> coeffZPerCell(nxyz, 0.0);
                    std::vector<int> countXPC(nxyz, 0);
                    std::vector<int> countYPC(nxyz, 0);
                    std::vector<int> countZPC(nxyz, 0);

                    for (gko::size_type row = 0; row < nRows; row++)
                    {
                        for (int j = hRowPtrs[row];
                             j < hRowPtrs[row+1]; j++)
                        {
                            int col = hColIdxs[j];
                            if (col == static_cast<int>(row)) continue;
                            int diff = std::abs(
                                col - static_cast<int>(row)
                            );
                            double val = std::abs(
                                static_cast<double>(hVals[j])
                            );
                            if (diff == 1)
                            {
                                coeffXPerCell[row] += val;
                                countXPC[row]++;
                            }
                            else if (diff == nx)
                            {
                                coeffYPerCell[row] += val;
                                countYPC[row]++;
                            }
                            else if (diff == nxy)
                            {
                                coeffZPerCell[row] += val;
                                countZPC[row]++;
                            }
                        }
                        if (countXPC[row] > 0)
                            coeffXPerCell[row] /= countXPC[row];
                        if (countYPC[row] > 0)
                            coeffYPerCell[row] /= countYPC[row];
                        if (countZPC[row] > 0)
                            coeffZPerCell[row] /= countZPC[row];
                    }

                    // Compute δr = r - r̄ and its statistics
                    double stdX = 0, stdY = 0, stdZ = 0;
                    double minCX = 1e30, maxCX = -1e30;
                    for (int i = 0; i < nxyz; i++)
                    {
                        double dX = coeffXPerCell[i] - meanCoeffX;
                        double dY = coeffYPerCell[i] - meanCoeffY;
                        double dZ = coeffZPerCell[i] - meanCoeffZ;
                        stdX += dX * dX;
                        stdY += dY * dY;
                        stdZ += dZ * dZ;
                        if (coeffXPerCell[i] < minCX)
                            minCX = coeffXPerCell[i];
                        if (coeffXPerCell[i] > maxCX)
                            maxCX = coeffXPerCell[i];
                    }
                    stdX = std::sqrt(stdX / nxyz);
                    stdY = std::sqrt(stdY / nxyz);
                    stdZ = std::sqrt(stdZ / nxyz);

                    double cvX = (meanCoeffX > 1e-30)
                        ? stdX / meanCoeffX : 0;
                    double cvY = (meanCoeffY > 1e-30)
                        ? stdY / meanCoeffY : 0;
                    double cvZ = (meanCoeffZ > 1e-30)
                        ? stdZ / meanCoeffZ : 0;

                    Info<< "FFT spectral: rAU variation:" << nl
                        << "  coeffX: mean=" << meanCoeffX
                        << " std=" << stdX
                        << " CV=" << cvX
                        << " range=[" << minCX << ", " << maxCX << "]"
                        << " ratio=" << (minCX > 1e-30
                            ? maxCX/minCX : 0)
                        << nl
                        << "  coeffY: CV=" << cvY << nl
                        << "  coeffZ: CV=" << cvZ << nl;

                    // 1D spectral analysis along x (averaged over y,z)
                    // For each y,z line, compute DCT of δr_x(x)
                    // Then average the spectral power |δr̂_m|²
                    std::vector<double> specPowerX(nx, 0.0);
                    for (int jy = 0; jy < ny; jy++)
                    {
                        for (int kz = 0; kz < nz; kz++)
                        {
                            // Extract x-line
                            std::vector<double> line(nx);
                            for (int ix = 0; ix < nx; ix++)
                            {
                                int idx = ix + jy*nx + kz*nxy;
                                line[ix] = coeffXPerCell[idx]
                                    - meanCoeffX;
                            }
                            // DCT-II (manual, O(N²) but nx=40 so fine)
                            for (int m = 0; m < nx; m++)
                            {
                                double sum = 0;
                                for (int ix = 0; ix < nx; ix++)
                                {
                                    sum += line[ix]
                                        * std::cos(
                                            M_PI * m * (ix + 0.5)
                                            / nx
                                          );
                                }
                                sum *= 2.0 / nx;  // DCT-II normalization
                                specPowerX[m] += sum * sum;
                            }
                        }
                    }
                    // Average over y,z lines
                    double totalPower = 0;
                    for (int m = 0; m < nx; m++)
                    {
                        specPowerX[m] /= (ny * nz);
                        totalPower += specPowerX[m];
                    }

                    // Print spectral power distribution
                    // (binned into low/mid/high frequency bands)
                    double lowPower = 0, midPower = 0, highPower = 0;
                    int lowEnd = nx / 8;      // modes 0..4 (λ/L > 0.25)
                    int midEnd = nx / 2;      // modes 5..19
                    for (int m = 0; m <= lowEnd; m++)
                        lowPower += specPowerX[m];
                    for (int m = lowEnd+1; m <= midEnd; m++)
                        midPower += specPowerX[m];
                    for (int m = midEnd+1; m < nx; m++)
                        highPower += specPowerX[m];

                    Info<< "  δr_x spectral power (DCT of x-lines):"
                        << nl
                        << "    total ||δr̂||² = " << totalPower << nl
                        << "    low  (m=0.." << lowEnd << "): "
                        << lowPower
                        << " (" << (totalPower > 1e-30
                            ? 100*lowPower/totalPower : 0)
                        << "%)" << nl
                        << "    mid  (m=" << lowEnd+1 << ".."
                        << midEnd << "): " << midPower
                        << " (" << (totalPower > 1e-30
                            ? 100*midPower/totalPower : 0)
                        << "%)" << nl
                        << "    high (m=" << midEnd+1 << ".."
                        << nx-1 << "): " << highPower
                        << " (" << (totalPower > 1e-30
                            ? 100*highPower/totalPower : 0)
                        << "%)" << nl;

                    // Print first 10 modes individually
                    Info<< "    modes: ";
                    for (int m = 0; m < std::min(nx, 10); m++)
                    {
                        Info<< "m" << m << "="
                            << specPowerX[m] << " ";
                    }
                    Info<< endl;

                    // 1D spectral analysis along Y
                    // For each x,z plane, compute DCT of δr_y(y)
                    std::vector<double> specPowerY(ny, 0.0);
                    for (int ix = 0; ix < nx; ix++)
                    {
                        for (int kz = 0; kz < nz; kz++)
                        {
                            std::vector<double> lineY(ny);
                            for (int jy = 0; jy < ny; jy++)
                            {
                                int idx = ix + jy*nx + kz*nxy;
                                lineY[jy] = coeffYPerCell[idx]
                                    - meanCoeffY;
                            }
                            for (int m = 0; m < ny; m++)
                            {
                                double sum = 0;
                                for (int jy = 0; jy < ny; jy++)
                                {
                                    sum += lineY[jy]
                                        * std::cos(
                                            M_PI * m * (jy + 0.5)
                                            / ny
                                          );
                                }
                                sum *= 2.0 / ny;
                                specPowerY[m] += sum * sum;
                            }
                        }
                    }
                    double totalPowerY = 0;
                    for (int m = 0; m < ny; m++)
                    {
                        specPowerY[m] /= (nx * nz);
                        totalPowerY += specPowerY[m];
                    }

                    Info<< "  δr_y spectral power (DCT of y-lines):"
                        << nl
                        << "    total ||δr̂_y||² = "
                        << totalPowerY << nl
                        << "    modes: ";
                    for (int m = 0; m < std::min(ny, 10); m++)
                    {
                        Info<< "m" << m << "="
                            << specPowerY[m] << " ";
                    }
                    Info<< endl;

                    // Compute "true x-variation" by subtracting
                    // per-line mean. This isolates pure x-modes.
                    std::vector<double> specPureX(nx, 0.0);
                    for (int jy = 0; jy < ny; jy++)
                    {
                        for (int kz = 0; kz < nz; kz++)
                        {
                            // Line mean
                            double lineMean = 0;
                            for (int ix = 0; ix < nx; ix++)
                            {
                                int idx = ix + jy*nx + kz*nxy;
                                lineMean += coeffXPerCell[idx];
                            }
                            lineMean /= nx;

                            // DCT of line minus its own mean
                            std::vector<double> lineD(nx);
                            for (int ix = 0; ix < nx; ix++)
                            {
                                int idx = ix + jy*nx + kz*nxy;
                                lineD[ix] = coeffXPerCell[idx]
                                    - lineMean;
                            }
                            for (int m = 1; m < nx; m++)
                            {
                                double sum = 0;
                                for (int ix = 0; ix < nx; ix++)
                                {
                                    sum += lineD[ix]
                                        * std::cos(
                                            M_PI * m * (ix + 0.5)
                                            / nx
                                          );
                                }
                                sum *= 2.0 / nx;
                                specPureX[m] += sum * sum;
                            }
                        }
                    }
                    double totalPureX = 0;
                    for (int m = 1; m < nx; m++)
                    {
                        specPureX[m] /= (ny * nz);
                        totalPureX += specPureX[m];
                    }

                    Info<< "  Pure x-variation (line mean removed):"
                        << nl
                        << "    total ||δr̂_x,pure||² = "
                        << totalPureX
                        << " (vs total " << totalPower
                        << ", pure x = "
                        << (totalPower > 1e-30
                            ? 100*totalPureX/totalPower : 0)
                        << "% of total δr)" << nl
                        << "    modes: ";
                    for (int m = 1; m < std::min(nx, 10); m++)
                    {
                        Info<< "m" << m << "="
                            << specPureX[m] << " ";
                    }
                    Info<< endl;

                    // BJ analysis
                    int bjCutoff = nx / (2 * blockSize_);
                    double bjBand = 0;
                    for (int m = bjCutoff; m < nx; m++)
                        bjBand += specPowerX[m];
                    double gapBand = 0;
                    for (int m = 1; m < bjCutoff; m++)
                        gapBand += specPowerX[m];

                    Info<< "  BJ analysis (blockSize="
                        << blockSize_ << "):" << nl
                        << "    BJ effective band (m>="
                        << bjCutoff << "): "
                        << bjBand
                        << " (" << (totalPower > 1e-30
                            ? 100*bjBand/totalPower : 0)
                        << "% of δr power)" << nl
                        << "    gap band (m=1.." << bjCutoff-1
                        << "): " << gapBand
                        << " (" << (totalPower > 1e-30
                            ? 100*gapBand/totalPower : 0)
                        << "% — NEITHER FFT nor BJ)" << nl
                        << endl;
                }

                // --- Ginkgo I/O Diagnostics ---
                if (debug_ >= 1)
                {
                    // A. CSR matrix structure verification
                    double minDiag = 1e30, maxDiag = -1e30;
                    double minRowSum = 1e30, maxRowSum = -1e30;
                    double sumRowSums = 0;
                    int unclassified = 0;

                    for (gko::size_type row = 0; row < nRows; row++)
                    {
                        double rowSum = 0;
                        double diagVal = 0;
                        for (int j = hRowPtrs[row]; j < hRowPtrs[row+1]; j++)
                        {
                            rowSum += static_cast<double>(hVals[j]);
                            if (hColIdxs[j] == static_cast<int>(row))
                            {
                                diagVal = static_cast<double>(hVals[j]);
                            }
                            else
                            {
                                int diff = std::abs(
                                    hColIdxs[j] - static_cast<int>(row)
                                );
                                if (diff != 1 && diff != nx && diff != nxy)
                                {
                                    unclassified++;
                                }
                            }
                        }
                        if (diagVal < minDiag) minDiag = diagVal;
                        if (diagVal > maxDiag) maxDiag = diagVal;
                        if (rowSum < minRowSum) minRowSum = rowSum;
                        if (rowSum > maxRowSum) maxRowSum = rowSum;
                        sumRowSums += rowSum;
                    }

                    double diagRatio = (std::abs(minDiag) > 1e-30)
                        ? std::abs(maxDiag) / std::abs(minDiag) : 0;

                    Info<< "FFT diag: CSR matrix diagnostics:" << nl
                        << "  diag range: [" << minDiag
                        << ", " << maxDiag << "]"
                        << (minDiag < 0 && maxDiag < 0
                            ? " (all negative=NSD OK)"
                            : " WARNING: not all negative!")
                        << nl
                        << "  diag ratio (max/min |diag|): "
                        << diagRatio
                        << (diagRatio < 2 ? " (uniform)"
                            : diagRatio < 10 ? " (mild variation)"
                            : " (LARGE variation)")
                        << nl
                        << "  row sum range: [" << minRowSum
                        << ", " << maxRowSum << "]"
                        << nl
                        << "  mean row sum: "
                        << sumRowSums / nRows
                        << nl
                        << "  unclassified off-diags: "
                        << unclassified
                        << (unclassified > 0
                            ? " WARNING: non-structured entries!"
                            : " (all structured)")
                        << nl
                        << "  sample diag[0]="
                        << static_cast<double>(hVals[hRowPtrs[0]])
                        << endl;

                    // B. FFT preconditioner I/O quality test
                    // Apply preconditioner to a test vector and check
                    using Dense = gko::matrix::Dense<ValueType>;

                    // Test with constant vector (null mode)
                    auto ones = Dense::create(
                        exec, gko::dim<2>{nCells, 1}
                    );
                    ones->fill(static_cast<ValueType>(1));
                    auto zOnes = Dense::create(
                        exec, gko::dim<2>{nCells, 1}
                    );
                    fftPrecond->apply(ones, zOnes);

                    // Copy to host and compute norm
                    std::vector<ValueType> hZOnes(nCells);
                    exec->get_master()->copy_from(
                        exec.get(), nCells,
                        zOnes->get_const_values(),
                        hZOnes.data()
                    );
                    double zOnesNorm = 0;
                    for (gko::size_type i = 0; i < nCells; i++)
                        zOnesNorm += double(hZOnes[i]) * double(hZOnes[i]);
                    zOnesNorm = std::sqrt(zOnesNorm);

                    Info<< "FFT diag: null mode test:"
                        << " ||FFT(ones)|| = " << zOnesNorm
                        << (zOnesNorm < 1e-6
                            ? " (PASS: null mode killed)"
                            : " (FAIL: null mode leaks!)")
                        << endl;

                    // Test with a random-ish vector (use diagonal as proxy)
                    auto testR = Dense::create(
                        exec, gko::dim<2>{nCells, 1}
                    );
                    // Fill with alternating pattern for a quick test
                    {
                        std::vector<ValueType> hTestR(nCells);
                        double testMean = 0;
                        for (gko::size_type i = 0; i < nCells; i++)
                        {
                            // Use first off-diag value as test input
                            hTestR[i] = static_cast<ValueType>(
                                std::sin(double(i) * 0.1)
                            );
                            testMean += double(hTestR[i]);
                        }
                        // Project out constant mode
                        testMean /= nCells;
                        for (gko::size_type i = 0; i < nCells; i++)
                            hTestR[i] -= static_cast<ValueType>(testMean);

                        exec->copy_from(
                            exec->get_master().get(), nCells,
                            hTestR.data(),
                            testR->get_values()
                        );
                    }

                    auto testZ = Dense::create(
                        exec, gko::dim<2>{nCells, 1}
                    );
                    fftPrecond->apply(testR, testZ);

                    // Compute A*z using CSR
                    auto testAz = Dense::create(
                        exec, gko::dim<2>{nCells, 1}
                    );
                    csrMtx->apply(testZ, testAz);

                    // Copy all to host for metrics
                    std::vector<ValueType> hR(nCells), hZ(nCells), hAz(nCells);
                    exec->get_master()->copy_from(
                        exec.get(), nCells,
                        testR->get_const_values(), hR.data()
                    );
                    exec->get_master()->copy_from(
                        exec.get(), nCells,
                        testZ->get_const_values(), hZ.data()
                    );
                    exec->get_master()->copy_from(
                        exec.get(), nCells,
                        testAz->get_const_values(), hAz.data()
                    );

                    double rNorm = 0, zNorm = 0;
                    double rDotZ = 0, zDotAz = 0;
                    double resNorm = 0;
                    for (gko::size_type i = 0; i < nCells; i++)
                    {
                        double ri = double(hR[i]);
                        double zi = double(hZ[i]);
                        double azi = double(hAz[i]);
                        rNorm += ri * ri;
                        zNorm += zi * zi;
                        rDotZ += ri * zi;
                        zDotAz += zi * azi;
                        double diff = azi - ri;
                        resNorm += diff * diff;
                    }
                    rNorm = std::sqrt(rNorm);
                    zNorm = std::sqrt(zNorm);
                    resNorm = std::sqrt(resNorm);

                    double relRes = (rNorm > 1e-30)
                        ? resNorm / rNorm : 0;
                    double alpha1 = (std::abs(zDotAz) > 1e-30)
                        ? rDotZ / zDotAz : 0;

                    Info<< "FFT diag: preconditioner quality:" << nl
                        << "  ||r|| = " << rNorm << nl
                        << "  ||z|| = " << zNorm << nl
                        << "  ||A*z - r|| / ||r|| = " << relRes
                        << (relRes < 0.01 ? " (EXCELLENT)"
                            : relRes < 0.5 ? " (GOOD)"
                            : relRes < 1.0 ? " (POOR)"
                            : " (HARMFUL: amplifies residual!)")
                        << nl
                        << "  r . z = " << rDotZ
                        << (rDotZ < 0
                            ? " (NEGATIVE=correct for NSD CG)"
                            : " (POSITIVE=WRONG SIGN for NSD!)")
                        << nl
                        << "  z . Az = " << zDotAz
                        << (zDotAz < 0
                            ? " (NEGATIVE=correct for NSD)"
                            : " (POSITIVE=wrong!)")
                        << nl
                        << "  alpha_1 = " << alpha1
                        << (alpha1 > 0
                            ? " (POSITIVE=CG OK)"
                            : " (NEGATIVE=CG WILL DIVERGE!)")
                        << nl
                        << "  r[0..4] = " << hR[0] << " " << hR[1]
                        << " " << hR[2] << " " << hR[3] << " " << hR[4]
                        << nl
                        << "  z[0..4] = " << hZ[0] << " " << hZ[1]
                        << " " << hZ[2] << " " << hZ[3] << " " << hZ[4]
                        << endl;
                }
            }

            if (preconditionerType_ == PreconditionerType::FFT_DIRECT)
            {
                // Store FFT preconditioner directly — Richardson loop
                // in solveImpl bypasses CG entirely
                solver = fftPrecond;
            }
            else if
            (
                preconditionerType_ == PreconditionerType::FFT_CHEBYSHEV
            )
            {
                // Chebyshev polynomial smoother wrapping FFT as inner
                // preconditioner, used as the CG preconditioner.
                // k Richardson steps with Chebyshev root-based relaxation
                // parameters for optimal polynomial smoothing of the
                // eigenvalue spectrum of FFT^{-1}*A.
                // Uses fixed scalars (no inner products) — works with NSD.
                auto chebyPrecond = gko::share(
                    ChebyshevPolyLinOp<ValueType>::create(
                        exec,
                        gko::dim<2>{nCells, nCells},
                        op->localMatrix(),
                        fftPrecond,
                        static_cast<int>(chebyDegree_),
                        static_cast<ValueType>(chebyEigMin_),
                        static_cast<ValueType>(chebyEigMax_)
                    )
                );

                if (debug_ >= 1)
                {
                    Info<< "FFT_CHEBYSHEV: degree=" << chebyDegree_
                        << " eigBounds=[" << chebyEigMin_ << ", "
                        << chebyEigMax_ << "]" << nl
                        << "  omegas: ";
                    // Recompute for printing (same formula as constructor)
                    double d = (chebyEigMax_ + chebyEigMin_) / 2.0;
                    double c = (chebyEigMax_ - chebyEigMin_) / 2.0;
                    for (label j = 0; j < chebyDegree_; j++)
                    {
                        double cosArg = std::cos(
                            M_PI * (2*j + 1.0) / (2.0*chebyDegree_)
                        );
                        double omega = 1.0 / (d - c * cosArg);
                        Info<< omega << " ";
                    }
                    Info<< nl
                        << "  (using GMRES: Chebyshev preconditioner"
                        << " is non-symmetric)" << endl;
                }

                // Use GMRES (not CG) because the Chebyshev polynomial
                // preconditioner p_k(L^{-1}A)*L^{-1} is NOT symmetric
                // when A has variable coefficients. Regular (not flexible)
                // GMRES is correct since the preconditioner is fixed.
                solver = gko::solver::Gmres<ValueType>::build()
                    .with_krylov_dim(200u)
                    .with_generated_preconditioner(chebyPrecond)
                    .with_criteria(iterCrit, resCrit)
                    .on(exec)
                    ->generate(op->localMatrix());
            }
            else if
            (
                preconditionerType_ == PreconditionerType::FFT_SCALED
            )
            {
                // Symmetric diagonal scaling: D^{-1/2} * FFT * D^{-1/2}
                // where D = diag(|a_ii| / meanAbsDiag).
                // This normalizes the variable rAU coefficient so the
                // FFT sees a more uniform-coefficient problem.
                // Symmetric sandwich preserves NSD for CG.

                // Extract diagonal from CSR
                auto csrMtx2 = gko::as<
                    gko::matrix::Csr<ValueType, int>
                >(op->localMatrix());
                auto nRows2 = csrMtx2->get_size()[0];
                auto rowPtrs2 = csrMtx2->get_const_row_ptrs();
                auto colIdxs2 = csrMtx2->get_const_col_idxs();
                auto vals2 = csrMtx2->get_const_values();

                std::vector<int> hRP(nRows2 + 1);
                std::vector<int> hCI(
                    csrMtx2->get_num_stored_elements()
                );
                std::vector<ValueType> hV(
                    csrMtx2->get_num_stored_elements()
                );

                exec->get_master()->copy_from(
                    exec.get(), nRows2 + 1, rowPtrs2, hRP.data()
                );
                exec->get_master()->copy_from(
                    exec.get(),
                    csrMtx2->get_num_stored_elements(),
                    colIdxs2, hCI.data()
                );
                exec->get_master()->copy_from(
                    exec.get(),
                    csrMtx2->get_num_stored_elements(),
                    vals2, hV.data()
                );

                double meanAbsDiag = 0;
                std::vector<double> absDiag(nCells, 0.0);
                for (gko::size_type row = 0; row < nRows2; row++)
                {
                    for (int j = hRP[row]; j < hRP[row+1]; j++)
                    {
                        if (hCI[j] == static_cast<int>(row))
                        {
                            absDiag[row] = std::abs(
                                static_cast<double>(hV[j])
                            );
                            meanAbsDiag += absDiag[row];
                            break;
                        }
                    }
                }
                meanAbsDiag /= nCells;

                // Build scale: sqrt(meanAbsDiag / |diag_i|)
                gko::array<ValueType> hScaleArr(
                    exec->get_master(),
                    static_cast<gko::size_type>(nCells)
                );
                auto scaleData = hScaleArr.get_data();
                double minScale = 1e30, maxScale = 0;
                for (gko::size_type i = 0; i < nCells; i++)
                {
                    double s = std::sqrt(
                        meanAbsDiag / std::max(absDiag[i], 1e-30)
                    );
                    scaleData[i] = static_cast<ValueType>(s);
                    if (s < minScale) minScale = s;
                    if (s > maxScale) maxScale = s;
                }

                if (debug_ >= 1)
                {
                    Info<< "FFT_SCALED: meanAbsDiag=" << meanAbsDiag
                        << " scale range=[" << minScale
                        << ", " << maxScale << "]"
                        << " ratio=" << maxScale / minScale
                        << endl;
                }

                // Copy to GPU and create Diagonal LinOp
                gko::array<ValueType> dScaleArr(exec, hScaleArr);
                auto diagScale = gko::share(
                    gko::matrix::Diagonal<ValueType>::create(
                        exec, nCells, std::move(dScaleArr)
                    )
                );

                // Compose: D^{-1/2} * FFT^{-1} * D^{-1/2}
                std::vector<std::shared_ptr<const gko::LinOp>> ops;
                ops.reserve(3);
                ops.push_back(diagScale);
                ops.push_back(fftPrecond);
                ops.push_back(diagScale);
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
            else
            {
                solver = gko::solver::Cg<ValueType>::build()
                    .with_generated_preconditioner(fftPrecond)
                    .with_criteria(iterCrit, resCrit)
                    .on(exec)
                    ->generate(op->localMatrix());
            }
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
                std::vector<int> hColIdxs(
                    csrMtx->get_num_stored_elements()
                );
                std::vector<ValueType> hVals(
                    csrMtx->get_num_stored_elements()
                );

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

                        int diff = std::abs(
                            col - static_cast<int>(row)
                        );
                        double val = std::abs(
                            static_cast<double>(hVals[j])
                        );

                        if (diff == 1)
                        {
                            sumCoeffX += val; countX++;
                        }
                        else if (diff == nx)
                        {
                            sumCoeffY += val; countY++;
                        }
                        else if (diff == nxy)
                        {
                            sumCoeffZ += val; countZ++;
                        }
                    }
                }

                double meanCoeffX = countX > 0 ? sumCoeffX/countX : 0;
                double meanCoeffY = countY > 0 ? sumCoeffY/countY : 0;
                double meanCoeffZ = countZ > 0 ? sumCoeffZ/countZ : 0;

                if (debug_ >= 1)
                {
                    Info<< "FFT+BJ: FFT coefficients:"
                        << " coeffX=" << meanCoeffX
                        << " coeffY=" << meanCoeffY
                        << " coeffZ=" << meanCoeffZ
                        << endl;
                }

                fft->updateCoeffs(meanCoeffX, meanCoeffY, meanCoeffZ);
            }

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
        else if
        (
            preconditionerType_ == PreconditionerType::FFT_BJ_VCYCLE
        )
        {
            // 2-level V-cycle: BJ-smooth + FFT-coarse + BJ-smooth
            // Symmetric by construction → compatible with CG.
            // BJ handles local/boundary modes, FFT handles global modes.
            auto nCells = static_cast<gko::size_type>(
                fftDimensions_.x() * fftDimensions_.y()
              * fftDimensions_.z()
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

            // Extract mean coupling coefficients and update FFT
            {
                auto csrMtx = gko::as<gko::matrix::Csr<ValueType, int>>(
                    op->localMatrix()
                );
                auto nRows = csrMtx->get_size()[0];
                auto rowPtrs = csrMtx->get_const_row_ptrs();
                auto colIdxs = csrMtx->get_const_col_idxs();
                auto vals = csrMtx->get_const_values();

                std::vector<int> hRowPtrs(nRows + 1);
                std::vector<int> hColIdxs(
                    csrMtx->get_num_stored_elements()
                );
                std::vector<ValueType> hVals(
                    csrMtx->get_num_stored_elements()
                );

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

                        int diff = std::abs(
                            col - static_cast<int>(row)
                        );
                        double val = std::abs(
                            static_cast<double>(hVals[j])
                        );

                        if (diff == 1)
                        {
                            sumCoeffX += val; countX++;
                        }
                        else if (diff == nx)
                        {
                            sumCoeffY += val; countY++;
                        }
                        else if (diff == nxy)
                        {
                            sumCoeffZ += val; countZ++;
                        }
                    }
                }

                double meanCoeffX = countX > 0 ? sumCoeffX/countX : 0;
                double meanCoeffY = countY > 0 ? sumCoeffY/countY : 0;
                double meanCoeffZ = countZ > 0 ? sumCoeffZ/countZ : 0;

                if (debug_ >= 1)
                {
                    Info<< "FFT_BJ_VCYCLE: FFT coefficients:"
                        << " coeffX=" << meanCoeffX
                        << " coeffY=" << meanCoeffY
                        << " coeffZ=" << meanCoeffZ
                        << endl;
                }

                fft->updateCoeffs(meanCoeffX, meanCoeffY, meanCoeffZ);
            }

            auto bj = gko::share(makeBJ());

            auto vcycle = gko::share(
                VCycleLinOp<ValueType>::create(
                    exec,
                    gko::dim<2>{nCells, nCells},
                    op->localMatrix(),
                    bj,
                    fft,
                    vcyclePreSmooth_,
                    vcyclePostSmooth_
                )
            );

            if (debug_ >= 1)
            {
                Info<< "FFT_BJ_VCYCLE: 2-level V-cycle"
                    << " preSmooth=" << vcyclePreSmooth_
                    << " postSmooth=" << vcyclePostSmooth_
                    << " blockSize=" << blockSize_ << endl;
            }

            solver = gko::solver::Cg<ValueType>::build()
                .with_generated_preconditioner(vcycle)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());
        }
        else if (
            preconditionerType_ == PreconditionerType::GEOMETRIC_MG_FFT
        )
        {
            // Geometric multigrid V-cycle with FFT coarsest-level solver.
            // Multi-level: Chebyshev(Jacobi) smoothers at each level,
            // geometric 2:1 coarsening, trilinear prolongation,
            // full-weighting restriction, FFT direct solve at coarsest.

            exec->synchronize();
            auto mgStart = std::chrono::high_resolution_clock::now();

            auto gmg = gko::share(
                GeometricMultigridLinOp<ValueType>::createHierarchy(
                    exec,
                    op->localMatrix(),
                    fftDimensions_.x(),
                    fftDimensions_.y(),
                    fftDimensions_.z(),
                    double(meshSpacing_.x()),
                    double(meshSpacing_.y()),
                    double(meshSpacing_.z()),
                    vcyclePreSmooth_,
                    vcyclePostSmooth_,
                    blockSize_,
                    std::string(vcycleSmoother_),
                    vcycleChebDegree_,
                    debug_
                )
            );

            exec->synchronize();
            auto mgEnd = std::chrono::high_resolution_clock::now();
            double mgSetupMs = std::chrono::duration<double, std::milli>(
                mgEnd - mgStart
            ).count();

            if (debug_ >= 1)
            {
                Info<< "GEOMETRIC_MG_FFT: hierarchy setup "
                    << mgSetupMs << " ms"
                    << ", smoother=" << vcycleSmoother_
                    << ", chebDegree=" << vcycleChebDegree_
                    << ", preSmooth=" << vcyclePreSmooth_
                    << ", postSmooth=" << vcyclePostSmooth_
                    << endl;
            }

            solver = gko::solver::Cg<ValueType>::build()
                .with_generated_preconditioner(gmg)
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
        else if (preconditionerType_ == PreconditionerType::MULTIGRID_FFT)
        {
            // Additive composite: FFT + AMG(1 V-cycle)
            //
            // The FFT captures structured (constant-coefficient) components
            // well, while AMG handles variable coefficients and boundary
            // effects. Their additive combination P = P_fft + P_mg is SPD
            // when both components are SPD (sum of SPD is SPD).

            // ---- FFT component ----
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

            // Extract mean coupling coefficients from CSR for FFT eigenvalues
            {
                auto csrMtx = gko::as<gko::matrix::Csr<ValueType, int>>(
                    op->localMatrix()
                );
                auto nRows = csrMtx->get_size()[0];
                auto rowPtrs = csrMtx->get_const_row_ptrs();
                auto colIdxs = csrMtx->get_const_col_idxs();
                auto vals = csrMtx->get_const_values();

                std::vector<int> hRowPtrs(nRows + 1);
                std::vector<int> hColIdxs(
                    csrMtx->get_num_stored_elements()
                );
                std::vector<ValueType> hVals(
                    csrMtx->get_num_stored_elements()
                );

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

                        int diff = std::abs(
                            col - static_cast<int>(row)
                        );
                        double val = std::abs(
                            static_cast<double>(hVals[j])
                        );

                        if (diff == 1)
                        {
                            sumCoeffX += val; countX++;
                        }
                        else if (diff == nx)
                        {
                            sumCoeffY += val; countY++;
                        }
                        else if (diff == nxy)
                        {
                            sumCoeffZ += val; countZ++;
                        }
                    }
                }

                double meanCoeffX = countX > 0 ? sumCoeffX/countX : 0;
                double meanCoeffY = countY > 0 ? sumCoeffY/countY : 0;
                double meanCoeffZ = countZ > 0 ? sumCoeffZ/countZ : 0;

                if (debug_ >= 1)
                {
                    Info<< "MG+FFT: FFT coefficients:"
                        << " coeffX=" << meanCoeffX
                        << " coeffY=" << meanCoeffY
                        << " coeffZ=" << meanCoeffZ
                        << endl;
                }

                fft->updateCoeffs(meanCoeffX, meanCoeffY, meanCoeffZ);
            }

            // ---- MG component (same as MULTIGRID path) ----
            std::shared_ptr<gko::LinOpFactory> smootherFactory;

            if (mgSmoother_ == "chebyshev")
            {
                constexpr ValueType chebyUpper =
                    static_cast<ValueType>(2.2);
                constexpr ValueType chebyLower =
                    chebyUpper / static_cast<ValueType>(30);

                auto jacInner = gko::share(
                    gko::preconditioner::Jacobi<ValueType, int>::build()
                        .with_max_block_size(1u)
                        .on(exec)
                );

                smootherFactory = gko::share(
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
            }
            else
            {
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

            auto pgmFactory = gko::share(
                gko::multigrid::Pgm<ValueType, int>::build()
                    .with_deterministic(true)
                    .on(exec)
            );

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

            // Generate MG hierarchy from matrix
            exec->synchronize();

            size_t vramFreeBefore = 0, vramTotal = 0;
            haloGetGpuMemInfo(&vramFreeBefore, &vramTotal);

            auto mgStart = std::chrono::high_resolution_clock::now();

            auto mgPrecond = gko::share(
                mgFactory->generate(op->localMatrix())
            );

            exec->synchronize();
            auto mgEnd = std::chrono::high_resolution_clock::now();

            size_t vramFreeAfter = 0, vramDummy = 0;
            haloGetGpuMemInfo(&vramFreeAfter, &vramDummy);
            long mgVramDelta =
                static_cast<long>(vramFreeBefore)
              - static_cast<long>(vramFreeAfter);

            double mgSetupMs = std::chrono::duration<double, std::milli>(
                mgEnd - mgStart
            ).count();

            if (debug_ >= 1)
            {
                Info<< "MG+FFT: MG hierarchy setup: "
                    << mgSetupMs << " ms, VRAM delta: "
                    << mgVramDelta / (1024*1024) << " MB" << endl;
            }

            // ---- Composite: FFT pre-smooth + MG defect correction ----
            // z = P_fft(r) + P_mg(r - A * P_fft(r))
            // FFT handles structured component, MG corrects variable/
            // boundary errors in the remaining defect.
            auto composite = gko::share(
                DefectCorrectionLinOp<ValueType>::create(
                    op->localMatrix(),  // system matrix A
                    fft,                // pre-smoother P1
                    mgPrecond           // corrector P2
                )
            );

            solver = gko::solver::Cg<ValueType>::build()
                .with_generated_preconditioner(composite)
                .with_criteria(iterCrit, resCrit)
                .on(exec)
                ->generate(op->localMatrix());
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
         && (   preconditionerType_ == PreconditionerType::MULTIGRID
             || preconditionerType_ == PreconditionerType::MULTIGRID_FFT)
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

        // FFT_DIRECT: Manual preconditioned CG with per-iteration logging.
        // Bypasses Ginkgo CG to diagnose spectral properties directly.
        // Uses the FFT Laplacian as the preconditioner M^{-1}.
        if (preconditionerType_ == PreconditionerType::FFT_DIRECT)
        {
            using Dense = gko::matrix::Dense<ValueType>;

            auto csrMtx = op->localMatrix();
            auto nCells = static_cast<gko::size_type>(
                fftDimensions_.x() * fftDimensions_.y()
              * fftDimensions_.z()
            );

            // "solver" is actually the FFT preconditioner
            auto& fftPrecond = solver;

            // Allocate CG work vectors
            auto r = Dense::create(exec, gko::dim<2>{nCells, 1});
            auto z = Dense::create(exec, gko::dim<2>{nCells, 1});
            auto p = Dense::create(exec, gko::dim<2>{nCells, 1});
            auto q = Dense::create(exec, gko::dim<2>{nCells, 1});
            auto normVec = Dense::create(
                exec->get_master(), gko::dim<2>{1, 1}
            );
            auto dotVec = Dense::create(
                exec->get_master(), gko::dim<2>{1, 1}
            );

            auto one = gko::initialize<Dense>({ValueType(1)}, exec);
            auto neg_one = gko::initialize<Dense>({ValueType(-1)}, exec);

            // r = b - A*x
            r->copy_from(b.get());
            csrMtx->apply(neg_one.get(), x.get(), one.get(), r.get());

            r->compute_norm2(normVec.get());
            double normR0 = static_cast<double>(
                normVec->get_const_values()[0]
            );

            // z = M^{-1}*r, p = z
            fftPrecond->apply(r.get(), z.get());
            p->copy_from(z.get());

            // rho = r^T * z (negative for NSD)
            r->compute_dot(z.get(), dotVec.get());
            double rho = static_cast<double>(
                dotVec->get_const_values()[0]
            );

            exec->synchronize();
            auto solveStart = std::chrono::high_resolution_clock::now();

            for (label iter = 0; iter < maxIter_; iter++)
            {
                // q = A*p
                csrMtx->apply(p.get(), q.get());

                // alpha = rho / (p^T * q)
                p->compute_dot(q.get(), dotVec.get());
                double pq = static_cast<double>(
                    dotVec->get_const_values()[0]
                );
                double alpha = (std::abs(pq) > 1e-30)
                    ? rho / pq : 0;

                // x += alpha * p
                auto alphaVec = gko::initialize<Dense>(
                    {ValueType(alpha)}, exec
                );
                x->add_scaled(alphaVec.get(), p.get());

                // r -= alpha * q
                auto negAlpha = gko::initialize<Dense>(
                    {ValueType(-alpha)}, exec
                );
                r->add_scaled(negAlpha.get(), q.get());

                // Check convergence
                r->compute_norm2(normVec.get());
                double normR = static_cast<double>(
                    normVec->get_const_values()[0]
                );
                double relRes = (normR0 > 1e-30)
                    ? normR / normR0 : 0;

                numIters = iter + 1;

                if (debug_ >= 1 && (iter < 10 || iter % 50 == 0
                    || relRes < static_cast<double>(tolerance)))
                {
                    Info<< "FFT_PCG: iter " << iter
                        << " alpha=" << alpha
                        << " ||r||/||r0||=" << relRes
                        << " rho=" << rho << endl;
                }

                if (relRes < static_cast<double>(tolerance))
                {
                    break;
                }

                // z = M^{-1}*r
                fftPrecond->apply(r.get(), z.get());

                // rho_new = r^T * z
                r->compute_dot(z.get(), dotVec.get());
                double rhoNew = static_cast<double>(
                    dotVec->get_const_values()[0]
                );

                // beta = rho_new / rho
                double beta = (std::abs(rho) > 1e-30)
                    ? rhoNew / rho : 0;

                // p = z + beta * p
                auto betaVec = gko::initialize<Dense>(
                    {ValueType(beta)}, exec
                );
                p->scale(betaVec.get());
                p->add_scaled(one.get(), z.get());

                rho = rhoNew;
            }

            OGLExecutor::instance().synchronize();
            auto solveEnd = std::chrono::high_resolution_clock::now();
            double solveMs = std::chrono::duration<double, std::milli>(
                solveEnd - solveStart
            ).count();

            Info<< "FFT_PCG: " << numIters << " iters, "
                << solveMs << " ms" << endl;

            // Copy solution back to OpenFOAM
            if constexpr (std::is_same<ValueType, float>::value)
            {
                FP32CastWrapper::fromGinkgoF32(x.get(), psi);
            }
            else
            {
                FP32CastWrapper::fromGinkgoF64(x.get(), psi);
            }

            return numIters;
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
        if
        (
            preconditionerType_ == PreconditionerType::MULTIGRID
         || preconditionerType_ == PreconditionerType::MULTIGRID_FFT
        )
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
            (   preconditionerType_ == PreconditionerType::MULTIGRID
             || preconditionerType_ == PreconditionerType::MULTIGRID_FFT)
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
            (   preconditionerType_ == PreconditionerType::MULTIGRID
             || preconditionerType_ == PreconditionerType::MULTIGRID_FFT)
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
                (   preconditionerType_ == PreconditionerType::MULTIGRID
                 || preconditionerType_ == PreconditionerType::MULTIGRID_FFT)
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
                    case PreconditionerType::FFT_DIRECT: return "FFT_DIRECT";
                    case PreconditionerType::FFT_BLOCK_JACOBI: return "fftBlockJacobi";
                    case PreconditionerType::FFT_CHEBYSHEV: return "fftChebyshev";
                    case PreconditionerType::FFT_SCALED: return "fftScaled";
                    case PreconditionerType::FFT_BJ_VCYCLE: return "fftBjVcycle";
                    case PreconditionerType::MULTIGRID: return "multigrid";
                    case PreconditionerType::MULTIGRID_FFT: return "multigridFFT";
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
