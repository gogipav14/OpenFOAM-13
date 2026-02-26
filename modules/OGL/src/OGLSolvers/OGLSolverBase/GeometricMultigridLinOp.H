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

Class
    Foam::OGL::GeometricMultigridLinOp

Description
    GPU geometric multigrid V-cycle preconditioner as a Ginkgo LinOp.

    Multi-level V-cycle with:
    - Geometric 2:1 coarsening (structured grids only)
    - Chebyshev(Jacobi) or Block Jacobi smoothers at each level
    - FFT direct solve (DCT/Neumann eigenvalues) at coarsest level
    - Trilinear prolongation with Neumann ghost clamping
    - Full-weighting restriction

    Preconditioner contract: apply(r, z) computes z = M^{-1} * r,
    approximately solving A*z = r from z=0. Symmetric when pre-smooth
    count == post-smooth count and smoother is symmetric. Compatible
    with CG for SPD pressure systems.

    Coarse-grid operators use per-cell coefficient restriction (NOT global
    mean) to track local coefficient variation (MRF zones, boundary effects).
    The coarsest-level FFT uses global mean coefficients (FFT only supports
    constant coefficients).

\*---------------------------------------------------------------------------*/

#ifndef GeometricMultigridLinOp_H
#define GeometricMultigridLinOp_H

#include "GinkgoCompat.H"
#include "FFTPreconditioner.H"
#include "MultigridKernels.h"
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

template <typename ValueType>
class GeometricMultigridLinOp
:
    public gko::EnableLinOp<GeometricMultigridLinOp<ValueType>>,
    public gko::EnableCreateMethod<GeometricMultigridLinOp<ValueType>>
{
    friend class gko::EnablePolymorphicObject<
        GeometricMultigridLinOp, gko::LinOp
    >;
    friend class gko::EnableCreateMethod<GeometricMultigridLinOp>;

public:

    using value_type = ValueType;
    using Dense = gko::matrix::Dense<ValueType>;
    using Csr = gko::matrix::Csr<ValueType, int>;

    // Destructor: free device coefficient arrays
    ~GeometricMultigridLinOp()
    {
        for (auto& level : levels_)
        {
            if (level.d_coeffX) cudaFree(level.d_coeffX);
            if (level.d_coeffY) cudaFree(level.d_coeffY);
            if (level.d_coeffZ) cudaFree(level.d_coeffZ);
        }
    }

    // Factory method: build the full multigrid hierarchy
    //
    // coarseSolverType: "fft" (DCT direct solve) or "chebyshev" (polynomial)
    // coarseChebDegree: degree for Chebyshev coarsest solver (only used if
    //                   coarseSolverType == "chebyshev", default 20)
    static std::shared_ptr<GeometricMultigridLinOp> createHierarchy(
        std::shared_ptr<const gko::Executor> exec,
        std::shared_ptr<const gko::LinOp> fineMatrix,
        int nx, int ny, int nz,
        double dx, double dy, double dz,
        int preSmooth, int postSmooth,
        int blockSize,
        const std::string& smootherType,
        int chebDegree,
        int debug,
        const std::string& coarseSolverType = "fft",
        int coarseChebDegree = 20
    )
    {
        auto nCells = static_cast<gko::size_type>(nx * ny * nz);

        auto mg = GeometricMultigridLinOp::create(
            exec, gko::dim<2>{nCells, nCells}
        );

        mg->preSmooth_ = preSmooth;
        mg->postSmooth_ = postSmooth;
        mg->debug_ = debug;

        // Determine number of levels by halving until coarse enough
        std::vector<int> dims_nx, dims_ny, dims_nz;
        int cnx = nx, cny = ny, cnz = nz;
        dims_nx.push_back(cnx);
        dims_ny.push_back(cny);
        dims_nz.push_back(cnz);

        while (true)
        {
            // Ceiling division: (n+1)/2 ensures every fine cell maps to
            // a coarse cell, eliminating orphan cells that break R ∝ P^T
            // symmetry required by CG.  The restriction kernel handles
            // partial blocks (count < 8) via boundary checks.
            int hnx = (cnx + 1) / 2;
            int hny = (cny + 1) / 2;
            int hnz = (cnz + 1) / 2;
            if (std::min({hnx, hny, hnz}) < 4 || hnx * hny * hnz < 100)
                break;
            cnx = hnx;
            cny = hny;
            cnz = hnz;
            dims_nx.push_back(cnx);
            dims_ny.push_back(cny);
            dims_nz.push_back(cnz);
        }

        int nLevels = static_cast<int>(dims_nx.size());

        if (debug >= 1)
        {
            fprintf(stderr,
                "GeometricMG: %d levels, finest %dx%dx%d (%d),"
                " coarsest %dx%dx%d (%d)\n",
                nLevels,
                dims_nx[0], dims_ny[0], dims_nz[0],
                dims_nx[0]*dims_ny[0]*dims_nz[0],
                dims_nx.back(), dims_ny.back(), dims_nz.back(),
                dims_nx.back()*dims_ny.back()*dims_nz.back());
        }

        // Extract per-cell directional coefficients from fine-level CSR
        auto csrMtx = gko::as<Csr>(fineMatrix);
        auto nRows = csrMtx->get_size()[0];
        auto nnzFine = csrMtx->get_num_stored_elements();

        // Copy CSR to host for coefficient extraction
        std::vector<int> hRowPtrs(nRows + 1);
        std::vector<int> hColIdxs(nnzFine);
        std::vector<ValueType> hVals(nnzFine);

        exec->get_master()->copy_from(
            exec.get(), nRows + 1,
            csrMtx->get_const_row_ptrs(), hRowPtrs.data()
        );
        exec->get_master()->copy_from(
            exec.get(), nnzFine,
            csrMtx->get_const_col_idxs(), hColIdxs.data()
        );
        exec->get_master()->copy_from(
            exec.get(), nnzFine,
            csrMtx->get_const_values(), hVals.data()
        );

        // Extract per-cell coefficients (host)
        std::vector<ValueType> hCoeffX(nCells, ValueType(0));
        std::vector<ValueType> hCoeffY(nCells, ValueType(0));
        std::vector<ValueType> hCoeffZ(nCells, ValueType(0));

        int fineNx = nx;
        int fineNxy = nx * ny;

        for (gko::size_type row = 0; row < nRows; row++)
        {
            for (int j = hRowPtrs[row]; j < hRowPtrs[row + 1]; j++)
            {
                int col = hColIdxs[j];
                if (col == static_cast<int>(row)) continue;

                int diff = std::abs(col - static_cast<int>(row));
                ValueType val = std::abs(hVals[j]);

                if (diff == 1)
                {
                    hCoeffX[row] += val;
                }
                else if (diff == fineNx)
                {
                    hCoeffY[row] += val;
                }
                else if (diff == fineNxy)
                {
                    hCoeffZ[row] += val;
                }
            }
            // Each cell has up to 2 neighbors per direction; average
            // The extracted value is the sum of absolute off-diag entries
            // in that direction. For a symmetric matrix, each direction
            // has 1-2 entries. We want the coupling coefficient per face.
            int ix = row % nx;
            int iy = (row / nx) % ny;

            int nxNeigh = (ix > 0 ? 1 : 0) + (ix < nx - 1 ? 1 : 0);
            int nyNeigh = (iy > 0 ? 1 : 0) + (iy < ny - 1 ? 1 : 0);
            int iz = row / (nx * ny);
            int nzNeigh = (iz > 0 ? 1 : 0) + (iz < nz - 1 ? 1 : 0);

            if (nxNeigh > 0) hCoeffX[row] /= ValueType(nxNeigh);
            if (nyNeigh > 0) hCoeffY[row] /= ValueType(nyNeigh);
            if (nzNeigh > 0) hCoeffZ[row] /= ValueType(nzNeigh);
        }

        if (debug >= 1)
        {
            // Print fine-level mean coefficients for reference
            double mX = 0, mY = 0, mZ = 0;
            for (gko::size_type i = 0; i < nCells; i++)
            {
                mX += hCoeffX[i]; mY += hCoeffY[i]; mZ += hCoeffZ[i];
            }
            mX /= nCells; mY /= nCells; mZ /= nCells;
            fprintf(stderr,
                "GeometricMG level 0: %dx%dx%d = %lu cells,"
                " mean coeffs=(%.4e,%.4e,%.4e)\n",
                nx, ny, nz, (unsigned long)nCells, mX, mY, mZ);
        }

        // Build levels
        mg->levels_.resize(nLevels);

        // Level 0 (finest): use the actual system matrix
        {
            auto& L0 = mg->levels_[0];
            L0.nx = nx;
            L0.ny = ny;
            L0.nz = nz;
            L0.nCells = nx * ny * nz;
            L0.matrix = gko::share(
                gko::clone(exec, csrMtx)
            );

            // Upload per-cell coefficients to GPU
            cudaMalloc(&L0.d_coeffX, nCells * sizeof(ValueType));
            cudaMalloc(&L0.d_coeffY, nCells * sizeof(ValueType));
            cudaMalloc(&L0.d_coeffZ, nCells * sizeof(ValueType));
            cudaMemcpy(L0.d_coeffX, hCoeffX.data(),
                nCells * sizeof(ValueType), cudaMemcpyHostToDevice);
            cudaMemcpy(L0.d_coeffY, hCoeffY.data(),
                nCells * sizeof(ValueType), cudaMemcpyHostToDevice);
            cudaMemcpy(L0.d_coeffZ, hCoeffZ.data(),
                nCells * sizeof(ValueType), cudaMemcpyHostToDevice);

            // Build smoother
            L0.smoother = buildSmoother(
                exec, L0.matrix, smootherType, blockSize, chebDegree
            );

            // Work vectors
            L0.rhs = Dense::create(exec, gko::dim<2>{nCells, 1});
            L0.e = Dense::create(exec, gko::dim<2>{nCells, 1});
            L0.r = Dense::create(exec, gko::dim<2>{nCells, 1});
            L0.z = Dense::create(exec, gko::dim<2>{nCells, 1});
        }

        // Coarser levels: restrict coefficients + build CSR + smoother
        for (int l = 1; l < nLevels; l++)
        {
            auto& Lc = mg->levels_[l];
            auto& Lf = mg->levels_[l - 1];
            Lc.nx = dims_nx[l];
            Lc.ny = dims_ny[l];
            Lc.nz = dims_nz[l];
            Lc.nCells = Lc.nx * Lc.ny * Lc.nz;

            // Allocate coarse coefficient arrays
            cudaMalloc(&Lc.d_coeffX, Lc.nCells * sizeof(ValueType));
            cudaMalloc(&Lc.d_coeffY, Lc.nCells * sizeof(ValueType));
            cudaMalloc(&Lc.d_coeffZ, Lc.nCells * sizeof(ValueType));

            // Restrict coefficients from fine to coarse level
            if constexpr (std::is_same_v<ValueType, float>)
            {
                mgRestrictCoeffsFloat(
                    Lf.d_coeffX, Lf.d_coeffY, Lf.d_coeffZ,
                    Lc.d_coeffX, Lc.d_coeffY, Lc.d_coeffZ,
                    Lf.nx, Lf.ny, Lf.nz,
                    Lc.nx, Lc.ny, Lc.nz,
                    0  // default stream during setup
                );
            }
            else
            {
                mgRestrictCoeffsDouble(
                    Lf.d_coeffX, Lf.d_coeffY, Lf.d_coeffZ,
                    Lc.d_coeffX, Lc.d_coeffY, Lc.d_coeffZ,
                    Lf.nx, Lf.ny, Lf.nz,
                    Lc.nx, Lc.ny, Lc.nz,
                    0
                );
            }
            cudaDeviceSynchronize();

            // Build 7-point CSR matrix from per-cell coefficients
            int nnz;
            if constexpr (std::is_same_v<ValueType, float>)
            {
                nnz = mgBuildCSR7ptFloat(
                    Lc.d_coeffX, Lc.d_coeffY, Lc.d_coeffZ,
                    Lc.nx, Lc.ny, Lc.nz,
                    nullptr, nullptr, nullptr, 0
                );
            }
            else
            {
                nnz = mgBuildCSR7ptDouble(
                    Lc.d_coeffX, Lc.d_coeffY, Lc.d_coeffZ,
                    Lc.nx, Lc.ny, Lc.nz,
                    nullptr, nullptr, nullptr, 0
                );
            }

            // Allocate CSR arrays as owned Ginkgo arrays (memory managed
            // by Ginkgo, freed when the CSR matrix is destroyed)
            auto rowPtrArr = gko::array<int>(exec, Lc.nCells + 1);
            auto colIdxArr = gko::array<int>(exec, nnz);
            auto valArr = gko::array<ValueType>(exec, nnz);

            // Get raw device pointers for our CUDA kernels
            int* d_rowPtr = rowPtrArr.get_data();
            int* d_colIdx = colIdxArr.get_data();
            ValueType* d_values = valArr.get_data();

            if constexpr (std::is_same_v<ValueType, float>)
            {
                mgBuildCSR7ptFloat(
                    Lc.d_coeffX, Lc.d_coeffY, Lc.d_coeffZ,
                    Lc.nx, Lc.ny, Lc.nz,
                    d_rowPtr, d_colIdx, d_values, 0
                );
            }
            else
            {
                mgBuildCSR7ptDouble(
                    Lc.d_coeffX, Lc.d_coeffY, Lc.d_coeffZ,
                    Lc.nx, Lc.ny, Lc.nz,
                    d_rowPtr, d_colIdx, d_values, 0
                );
            }
            cudaDeviceSynchronize();

            auto coarseDim = gko::dim<2>{
                static_cast<gko::size_type>(Lc.nCells),
                static_cast<gko::size_type>(Lc.nCells)
            };

            Lc.matrix = gko::share(
                Csr::create(
                    exec, coarseDim,
                    std::move(valArr),
                    std::move(colIdxArr),
                    std::move(rowPtrArr)
                )
            );

            // Build smoother for this level
            Lc.smoother = buildSmoother(
                exec, Lc.matrix, smootherType, blockSize, chebDegree
            );

            // Work vectors
            auto nc = static_cast<gko::size_type>(Lc.nCells);
            Lc.rhs = Dense::create(exec, gko::dim<2>{nc, 1});
            Lc.e = Dense::create(exec, gko::dim<2>{nc, 1});
            Lc.r = Dense::create(exec, gko::dim<2>{nc, 1});
            Lc.z = Dense::create(exec, gko::dim<2>{nc, 1});

            if (debug >= 1)
            {
                // Print mean coefficient values for this level
                std::vector<ValueType> dbgX(Lc.nCells), dbgY(Lc.nCells),
                    dbgZ(Lc.nCells);
                cudaMemcpy(dbgX.data(), Lc.d_coeffX,
                    Lc.nCells * sizeof(ValueType), cudaMemcpyDeviceToHost);
                cudaMemcpy(dbgY.data(), Lc.d_coeffY,
                    Lc.nCells * sizeof(ValueType), cudaMemcpyDeviceToHost);
                cudaMemcpy(dbgZ.data(), Lc.d_coeffZ,
                    Lc.nCells * sizeof(ValueType), cudaMemcpyDeviceToHost);
                double mX = 0, mY = 0, mZ = 0;
                for (int i = 0; i < Lc.nCells; i++)
                {
                    mX += dbgX[i]; mY += dbgY[i]; mZ += dbgZ[i];
                }
                mX /= Lc.nCells; mY /= Lc.nCells; mZ /= Lc.nCells;
                fprintf(stderr,
                    "GeometricMG level %d: %dx%dx%d = %d cells,"
                    " nnz=%d, mean coeffs=(%.4e,%.4e,%.4e)\n",
                    l, Lc.nx, Lc.ny, Lc.nz, Lc.nCells, nnz,
                    mX, mY, mZ);
            }
        }

        // Build coarsest-level solver (FFT or Chebyshev)
        {
            auto& Lcoarsest = mg->levels_.back();
            int cNx = Lcoarsest.nx;
            int cNy = Lcoarsest.ny;
            int cNz = Lcoarsest.nz;
            auto cN = static_cast<gko::size_type>(Lcoarsest.nCells);

            if (coarseSolverType == "chebyshev")
            {
                // Chebyshev polynomial solver at coarsest level.
                // Higher degree (default 20) for an approximate direct solve.
                // Uses the coarsest-level CSR matrix (already built above).
                auto& csrCoarsest = Lcoarsest.matrix;

                constexpr ValueType chebyUpper =
                    static_cast<ValueType>(2.2);
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
                                        coarseChebDegree
                                    )
                                )
                                .on(exec)
                        )
                        .on(exec)
                );

                mg->coarseSolver_ = gko::share(
                    chebyFactory->generate(csrCoarsest)
                );

                if (debug >= 1)
                {
                    fprintf(stderr,
                        "GeometricMG coarsest Chebyshev: %dx%dx%d,"
                        " degree=%d\n",
                        cNx, cNy, cNz, coarseChebDegree);
                }
            }
            else
            {
                // FFT direct solve at coarsest level

                // Compute mean coefficients at coarsest level for FFT
                std::vector<ValueType> hcX(Lcoarsest.nCells);
                std::vector<ValueType> hcY(Lcoarsest.nCells);
                std::vector<ValueType> hcZ(Lcoarsest.nCells);
                cudaMemcpy(hcX.data(), Lcoarsest.d_coeffX,
                    Lcoarsest.nCells * sizeof(ValueType),
                    cudaMemcpyDeviceToHost);
                cudaMemcpy(hcY.data(), Lcoarsest.d_coeffY,
                    Lcoarsest.nCells * sizeof(ValueType),
                    cudaMemcpyDeviceToHost);
                cudaMemcpy(hcZ.data(), Lcoarsest.d_coeffZ,
                    Lcoarsest.nCells * sizeof(ValueType),
                    cudaMemcpyDeviceToHost);

                double meanX = 0, meanY = 0, meanZ = 0;
                for (int i = 0; i < Lcoarsest.nCells; i++)
                {
                    meanX += static_cast<double>(hcX[i]);
                    meanY += static_cast<double>(hcY[i]);
                    meanZ += static_cast<double>(hcZ[i]);
                }
                meanX /= Lcoarsest.nCells;
                meanY /= Lcoarsest.nCells;
                meanZ /= Lcoarsest.nCells;

                // Compute coarsest-level mesh spacing from domain length.
                // With ceiling division, coarsening factor isn't exactly 2^n,
                // so derive spacing from actual dimensions: cdx = Lx / cnx.
                double cdx = (nx * dx) / static_cast<double>(cNx);
                double cdy = (ny * dy) / static_cast<double>(cNy);
                double cdz = (nz * dz) / static_cast<double>(cNz);

                mg->coarseSolver_ = gko::share(
                    FFTPreconditioner<ValueType>::create(
                        exec,
                        gko::dim<2>{cN, cN},
                        cNx, cNy, cNz,
                        cdx, cdy, cdz
                    )
                );

                // Update FFT eigenvalues with actual coarsest-level
                // coefficients.
                auto fftPtr = std::dynamic_pointer_cast<
                    FFTPreconditioner<ValueType>
                >(
                    std::const_pointer_cast<gko::LinOp>(mg->coarseSolver_)
                );
                if (fftPtr)
                {
                    fftPtr->updateCoeffs(meanX, meanY, meanZ);

                    if (debug >= 1)
                    {
                        fprintf(stderr,
                            "GeometricMG FFT coeffs:"
                            " mean=(%.4e,%.4e,%.4e)\n",
                            meanX, meanY, meanZ);
                    }
                }

                if (debug >= 1)
                {
                    fprintf(stderr,
                        "GeometricMG coarsest FFT: %dx%dx%d,"
                        " dx=%.6f dy=%.6f dz=%.6f\n",
                        cNx, cNy, cNz, cdx, cdy, cdz);
                }
            }
        }

        // Pre-allocate scalar constants
        mg->one_ = gko::initialize<Dense>({ValueType(1)}, exec);
        mg->neg_one_ = gko::initialize<Dense>({ValueType(-1)}, exec);

        return mg;
    }


protected:

    // Default constructor (required by EnableCreateMethod)
    explicit GeometricMultigridLinOp(
        std::shared_ptr<const gko::Executor> exec
    )
    :
        gko::EnableLinOp<GeometricMultigridLinOp>(exec),
        preSmooth_(1),
        postSmooth_(1),
        debug_(0)
    {}

    // Size constructor (used by create())
    GeometricMultigridLinOp(
        std::shared_ptr<const gko::Executor> exec,
        gko::dim<2> size
    )
    :
        gko::EnableLinOp<GeometricMultigridLinOp>(exec, size),
        preSmooth_(1),
        postSmooth_(1),
        debug_(0)
    {}

    // V-cycle apply: z = M^{-1} * r (from z=0)
    //
    // Standard multigrid V-cycle as preconditioner:
    //   Level 0 (finest): x is the output, b is the input RHS
    //   Level l (l>0):    e_l is the error, rhs_l is the restricted residual
    //
    // Each level preserves its RHS in the `rhs` vector for post-smooth.
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto exec = this->get_executor();
        auto dense_b = gko::as<Dense>(b);
        auto dense_x = gko::as<Dense>(x);

        int nLevels = static_cast<int>(levels_.size());

        // x = 0
        dense_x->fill(ValueType(0));

        if (nLevels == 1)
        {
            // Single level: just apply coarse FFT solve
            coarseSolver_->apply(dense_b, dense_x);
            return;
        }

        // Helper lambda for norms (debug only)
        auto computeNorm = [&](const Dense* v) -> double {
            auto normVec = Dense::create(
                this->get_executor()->get_master(), gko::dim<2>{1, 1}
            );
            const_cast<Dense*>(v)->compute_norm2(normVec.get());
            return static_cast<double>(normVec->get_const_values()[0]);
        };

        bool trace = (callCount_ == 0 && debug_ >= 2);

        if (trace)
            fprintf(stderr, "  [MG] ||b||=%.4e\n", computeNorm(dense_b));

        // === Downward leg: pre-smooth and restrict ===

        // Level 0: pre-smooth on x with RHS = b (external, not stored)
        // smoother->apply(b, x) computes x = M^{-1} * b from x=0
        levels_[0].smoother->apply(dense_b, dense_x);
        for (int s = 1; s < preSmooth_; s++)
        {
            // r = b - A*x
            levels_[0].r->copy_from(dense_b);
            levels_[0].matrix->apply(
                neg_one_.get(), dense_x, one_.get(), levels_[0].r.get()
            );
            levels_[0].z->fill(ValueType(0));
            levels_[0].smoother->apply(
                levels_[0].r.get(), levels_[0].z.get()
            );
            dense_x->add_scaled(one_.get(), levels_[0].z.get());
        }

        // Residual at level 0: r = b - A*x
        levels_[0].r->copy_from(dense_b);
        levels_[0].matrix->apply(
            neg_one_.get(), dense_x, one_.get(), levels_[0].r.get()
        );

        if (trace)
            fprintf(stderr, "  [MG] after L0 pre-smooth: ||r0||=%.4e\n",
                computeNorm(levels_[0].r.get()));

        // Restrict residual to level 1's RHS
        restrictLevel(0, levels_[0].r.get(), levels_[1].rhs.get());

        // Intermediate levels: pre-smooth, compute residual, restrict
        for (int l = 1; l < nLevels - 1; l++)
        {
            // Pre-smooth from zero: e_l = smooth(A_l, rhs_l, 0)
            levels_[l].e->fill(ValueType(0));
            levels_[l].smoother->apply(
                levels_[l].rhs.get(), levels_[l].e.get()
            );
            for (int s = 1; s < preSmooth_; s++)
            {
                // r = rhs - A*e
                levels_[l].r->copy_from(levels_[l].rhs.get());
                levels_[l].matrix->apply(
                    neg_one_.get(), levels_[l].e.get(),
                    one_.get(), levels_[l].r.get()
                );
                levels_[l].z->fill(ValueType(0));
                levels_[l].smoother->apply(
                    levels_[l].r.get(), levels_[l].z.get()
                );
                levels_[l].e->add_scaled(one_.get(), levels_[l].z.get());
            }

            // Residual: r = rhs_l - A_l * e_l
            levels_[l].r->copy_from(levels_[l].rhs.get());
            levels_[l].matrix->apply(
                neg_one_.get(), levels_[l].e.get(),
                one_.get(), levels_[l].r.get()
            );

            // Restrict residual to next coarser level's RHS
            restrictLevel(
                l, levels_[l].r.get(), levels_[l + 1].rhs.get()
            );

            if (trace)
                fprintf(stderr, "  [MG] L%d pre-smooth: ||rhs||=%.4e"
                    " ||e||=%.4e ||r||=%.4e\n",
                    l, computeNorm(levels_[l].rhs.get()),
                    computeNorm(levels_[l].e.get()),
                    computeNorm(levels_[l].r.get()));
        }

        // === Coarsest level: FFT direct solve ===
        {
            int cl = nLevels - 1;

            if (trace)
                fprintf(stderr, "  [MG] coarsest rhs: ||rhs_%d||=%.4e\n",
                    cl, computeNorm(levels_[cl].rhs.get()));

            coarseSolver_->apply(
                levels_[cl].rhs.get(), levels_[cl].e.get()
            );

            if (trace)
                fprintf(stderr, "  [MG] coarsest solve: ||e_%d||=%.4e\n",
                    cl, computeNorm(levels_[cl].e.get()));
        }

        // === Upward leg: prolong and post-smooth ===
        for (int l = nLevels - 2; l >= 1; l--)
        {
            // Prolong coarser correction and add
            prolongLevel(l, levels_[l + 1].e.get(), levels_[l].z.get());
            levels_[l].e->add_scaled(one_.get(), levels_[l].z.get());

            // Post-smooth: improve e_l given rhs_l (which is preserved)
            for (int s = 0; s < postSmooth_; s++)
            {
                // r = rhs_l - A_l * e_l
                levels_[l].r->copy_from(levels_[l].rhs.get());
                levels_[l].matrix->apply(
                    neg_one_.get(), levels_[l].e.get(),
                    one_.get(), levels_[l].r.get()
                );
                levels_[l].z->fill(ValueType(0));
                levels_[l].smoother->apply(
                    levels_[l].r.get(), levels_[l].z.get()
                );
                levels_[l].e->add_scaled(one_.get(), levels_[l].z.get());
            }
        }

        // Prolong level 1 correction back to level 0
        prolongLevel(0, levels_[1].e.get(), levels_[0].z.get());
        dense_x->add_scaled(one_.get(), levels_[0].z.get());

        if (trace)
        {
            // Compute residual after coarse correction
            levels_[0].r->copy_from(dense_b);
            levels_[0].matrix->apply(
                neg_one_.get(), dense_x, one_.get(), levels_[0].r.get()
            );
            fprintf(stderr, "  [MG] after coarse correction: ||r0||=%.4e\n",
                computeNorm(levels_[0].r.get()));
        }

        // Post-smooth at level 0: improve x given b (external RHS)
        for (int s = 0; s < postSmooth_; s++)
        {
            // r = b - A*x
            levels_[0].r->copy_from(dense_b);
            levels_[0].matrix->apply(
                neg_one_.get(), dense_x, one_.get(), levels_[0].r.get()
            );
            levels_[0].z->fill(ValueType(0));
            levels_[0].smoother->apply(
                levels_[0].r.get(), levels_[0].z.get()
            );
            dense_x->add_scaled(one_.get(), levels_[0].z.get());
        }

        // One-shot diagnostic
        if (callCount_ == 0 && debug_ >= 1)
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
            levels_[0].r->copy_from(dense_b);
            levels_[0].matrix->apply(
                neg_one_.get(), dense_x, one_.get(), levels_[0].r.get()
            );
            levels_[0].r->compute_norm2(normVec.get());
            double rNorm = static_cast<double>(
                normVec->get_const_values()[0]
            );
            fprintf(stderr,
                "GeometricMG V-cycle: ||b||=%.4e ||x||=%.4e"
                " ||b-Ax||=%.4e reduction=%.4f\n",
                bNorm, xNorm, rNorm,
                (bNorm > 1e-30) ? rNorm / bNorm : 0.0);
        }
        callCount_++;
    }

    void apply_impl(
        const gko::LinOp* alpha,
        const gko::LinOp* b,
        const gko::LinOp* beta,
        gko::LinOp* x
    ) const override
    {
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

    // Per-level data
    struct Level
    {
        int nx, ny, nz, nCells;
        std::shared_ptr<Csr> matrix;
        std::shared_ptr<const gko::LinOp> smoother;
        ValueType* d_coeffX = nullptr;
        ValueType* d_coeffY = nullptr;
        ValueType* d_coeffZ = nullptr;
        // Work vectors:
        //   rhs - right-hand side (preserved across pre/post smooth)
        //   e   - error/correction being computed
        //   r   - temp residual / workspace
        //   z   - temp workspace
        mutable std::shared_ptr<Dense> rhs, e, r, z;
    };

    std::vector<Level> levels_;
    std::shared_ptr<const gko::LinOp> coarseSolver_;  // FFT at coarsest
    int preSmooth_;
    int postSmooth_;
    int debug_;

    mutable std::shared_ptr<Dense> one_;
    mutable std::shared_ptr<Dense> neg_one_;
    mutable int callCount_ = 0;


    // Restrict from level l to level l+1
    void restrictLevel(int l, const Dense* fineVec, Dense* coarseVec) const
    {
        auto& Lf = levels_[l];
        auto& Lc = levels_[l + 1];
        const auto* fPtr = fineVec->get_const_values();
        auto* cPtr = coarseVec->get_values();

        if constexpr (std::is_same_v<ValueType, float>)
        {
            mgRestrict3DFloat(fPtr, cPtr,
                Lf.nx, Lf.ny, Lf.nz,
                Lc.nx, Lc.ny, Lc.nz, 0);
        }
        else
        {
            mgRestrict3DDouble(fPtr, cPtr,
                Lf.nx, Lf.ny, Lf.nz,
                Lc.nx, Lc.ny, Lc.nz, 0);
        }
    }

    // Prolong from level l+1 to level l (result written to fineVec)
    void prolongLevel(int l, const Dense* coarseVec, Dense* fineVec) const
    {
        auto& Lf = levels_[l];
        auto& Lc = levels_[l + 1];
        const auto* cPtr = coarseVec->get_const_values();
        auto* fPtr = fineVec->get_values();

        if constexpr (std::is_same_v<ValueType, float>)
        {
            mgProlong3DFloat(cPtr, fPtr,
                Lf.nx, Lf.ny, Lf.nz,
                Lc.nx, Lc.ny, Lc.nz, 0);
        }
        else
        {
            mgProlong3DDouble(cPtr, fPtr,
                Lf.nx, Lf.ny, Lf.nz,
                Lc.nx, Lc.ny, Lc.nz, 0);
        }
    }

    // Build smoother LinOp for a given level's matrix
    static std::shared_ptr<const gko::LinOp> buildSmoother(
        std::shared_ptr<const gko::Executor> exec,
        std::shared_ptr<Csr> matrix,
        const std::string& type,
        int blockSize,
        int chebDegree
    )
    {
        if (type == "chebyshev")
        {
            // Chebyshev(Jacobi) polynomial smoother.
            // Fixed foci for SPD M-matrices: [0.073, 2.2]
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
                                static_cast<gko::size_type>(chebDegree)
                            )
                            .on(exec)
                    )
                    .on(exec)
            );

            return gko::share(chebyFactory->generate(matrix));
        }
        else
        {
            // Block Jacobi smoother
            auto bjFactory = gko::share(
                gko::preconditioner::Jacobi<ValueType, int>::build()
                    .with_max_block_size(
                        static_cast<unsigned>(blockSize)
                    )
                    .on(exec)
            );

            return gko::share(bjFactory->generate(matrix));
        }
    }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
