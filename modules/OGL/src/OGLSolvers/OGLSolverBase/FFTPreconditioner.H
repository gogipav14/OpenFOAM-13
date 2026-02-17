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
    Foam::OGL::FFTPreconditioner

Description
    Ginkgo LinOp wrapping the cuFFT-based Laplacian preconditioner.

    Applies an approximate inverse of the discrete Laplacian using FFT
    diagonalization on uniform Cartesian grids. The preconditioner
    exploits that the periodic Laplacian is diagonalized by the DFT:

        L = F^{-1} * diag(lambda) * F

    so the preconditioner application is:

        M^{-1} * r = F^{-1} * (F(r) ./ lambda)

    at O(N log N) cost via cuFFT.

    For non-periodic BCs (e.g. walls with Neumann pressure), the
    periodic eigenvalues serve as an approximation. The Krylov solver
    (CG) handles the boundary mismatch.

    Can be used standalone or composed with Block Jacobi via
    AdditiveLinOp for hybrid FFT+BJ preconditioning.

    Reference: moljax paper, Section 3.4 (FFT-Diagonalized Operators)

\*---------------------------------------------------------------------------*/

#ifndef FFTPreconditioner_H
#define FFTPreconditioner_H

#include "GinkgoCompat.H"
#include "FFTKernels.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

template <typename ValueType>
class FFTPreconditioner
:
    public gko::EnableLinOp<FFTPreconditioner<ValueType>>,
    public gko::EnableCreateMethod<FFTPreconditioner<ValueType>>
{
    friend class gko::EnablePolymorphicObject<FFTPreconditioner, gko::LinOp>;
    friend class gko::EnableCreateMethod<FFTPreconditioner>;

public:

    using value_type = ValueType;

    // Destructor must be public for shared_ptr/unique_ptr deletion
    ~FFTPreconditioner()
    {
        if (handle_)
        {
            fftPrecondDestroy(handle_);
            handle_ = nullptr;
        }
    }


protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto dense_b =
            gko::as<gko::matrix::Dense<ValueType>>(b);
        auto dense_x =
            gko::as<gko::matrix::Dense<ValueType>>(x);

        // Get raw GPU pointers from Ginkgo Dense vectors
        const ValueType* b_ptr = dense_b->get_const_values();
        ValueType* x_ptr = dense_x->get_values();

        // Dispatch to correct precision
        if constexpr (std::is_same<ValueType, float>::value)
        {
            fftPrecondApplyFloat(handle_, b_ptr, x_ptr, totalCells_);
        }
        else
        {
            fftPrecondApplyDouble(handle_, b_ptr, x_ptr, totalCells_);
        }
    }

    void apply_impl
    (
        const gko::LinOp* alpha,
        const gko::LinOp* b,
        const gko::LinOp* beta,
        gko::LinOp* x
    ) const override
    {
        auto dense_x =
            gko::as<gko::matrix::Dense<ValueType>>(x);

        // temp = M^{-1} * b
        auto temp = gko::matrix::Dense<ValueType>::create(
            this->get_executor(), dense_x->get_size()
        );
        this->apply_impl(b, temp.get());

        // x = alpha * temp + beta * x
        auto dense_alpha =
            gko::as<gko::matrix::Dense<ValueType>>(alpha);
        auto dense_beta =
            gko::as<gko::matrix::Dense<ValueType>>(beta);
        dense_x->scale(dense_beta);
        dense_x->add_scaled(dense_alpha, temp);
    }

    // Default constructor (required by EnableCreateMethod)
    explicit FFTPreconditioner(std::shared_ptr<const gko::Executor> exec)
    :
        gko::EnableLinOp<FFTPreconditioner>(exec),
        handle_(nullptr),
        totalCells_(0)
    {}

    // Construct with grid dimensions
    FFTPreconditioner
    (
        std::shared_ptr<const gko::Executor> exec,
        gko::dim<2> size,
        int nx, int ny, int nz,
        double dx, double dy, double dz
    )
    :
        gko::EnableLinOp<FFTPreconditioner>(exec, size),
        totalCells_(nx * ny * nz)
    {
        int useFloat = std::is_same<ValueType, float>::value ? 1 : 0;
        handle_ = fftPrecondCreate(nx, ny, nz, dx, dy, dz, useFloat);
    }


public:

    // Update eigenvalues with actual matrix coupling coefficients
    void updateCoeffs(double coeffX, double coeffY, double coeffZ)
    {
        if (handle_)
        {
            fftPrecondUpdateCoeffs(handle_, coeffX, coeffY, coeffZ);
        }
    }


private:

    FFTPrecondHandle handle_;
    int totalCells_;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
