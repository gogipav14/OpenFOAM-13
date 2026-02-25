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
    Foam::OGL::CylFFTPreconditioner

Description
    Ginkgo LinOp wrapping the cylindrical FFT+Thomas preconditioner.

    Applies an approximate inverse of the discrete Laplacian on structured
    cylindrical (r, theta) grids. The theta direction is decoupled by DFT
    (periodic), leaving independent tridiagonal systems in r solved via
    pre-factored Thomas algorithm.

    Data layout: r-major, theta varies fastest.
      data[i_r * ntheta + i_theta]

    The Krylov solver (CG/PCG) handles variable coefficient mismatch.

\*---------------------------------------------------------------------------*/

#ifndef CylFFTPreconditioner_H
#define CylFFTPreconditioner_H

#include "GinkgoCompat.H"
#include "CylFFTKernels.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

template <typename ValueType>
class CylFFTPreconditioner
:
    public gko::EnableLinOp<CylFFTPreconditioner<ValueType>>,
    public gko::EnableCreateMethod<CylFFTPreconditioner<ValueType>>
{
    friend class gko::EnablePolymorphicObject<CylFFTPreconditioner, gko::LinOp>;
    friend class gko::EnableCreateMethod<CylFFTPreconditioner>;

public:

    using value_type = ValueType;

    ~CylFFTPreconditioner()
    {
        if (handle_)
        {
            cylFFTPrecondDestroy(handle_);
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

        const ValueType* b_ptr = dense_b->get_const_values();
        ValueType* x_ptr = dense_x->get_values();

        if (sectorMode_)
        {
            if constexpr (std::is_same<ValueType, float>::value)
            {
                cylFFTPrecondApplySectorFloat(
                    handle_, b_ptr, x_ptr, totalCells_);
            }
            else
            {
                cylFFTPrecondApplySectorDouble(
                    handle_, b_ptr, x_ptr, totalCells_);
            }
        }
        else
        {
            if constexpr (std::is_same<ValueType, float>::value)
            {
                cylFFTPrecondApplyFloat(
                    handle_, b_ptr, x_ptr, totalCells_);
            }
            else
            {
                cylFFTPrecondApplyDouble(
                    handle_, b_ptr, x_ptr, totalCells_);
            }
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

        auto temp = gko::matrix::Dense<ValueType>::create(
            this->get_executor(), dense_x->get_size()
        );
        this->apply_impl(b, temp.get());

        auto dense_alpha =
            gko::as<gko::matrix::Dense<ValueType>>(alpha);
        auto dense_beta =
            gko::as<gko::matrix::Dense<ValueType>>(beta);
        dense_x->scale(dense_beta);
        dense_x->add_scaled(dense_alpha, temp);
    }

    // Default constructor (required by EnableCreateMethod)
    explicit CylFFTPreconditioner(std::shared_ptr<const gko::Executor> exec)
    :
        gko::EnableLinOp<CylFFTPreconditioner>(exec),
        handle_(nullptr),
        totalCells_(0),
        sectorMode_(false)
    {}

    // Construct with cylindrical grid dimensions (DFT mode)
    CylFFTPreconditioner
    (
        std::shared_ptr<const gko::Executor> exec,
        gko::dim<2> size,
        int nr, int ntheta
    )
    :
        gko::EnableLinOp<CylFFTPreconditioner>(exec, size),
        totalCells_(nr * ntheta),
        sectorMode_(false)
    {
        int useFloat = std::is_same<ValueType, float>::value ? 1 : 0;
        handle_ = cylFFTPrecondCreate(nr, ntheta, useFloat);
    }

    // Construct with per-sector DCT dimensions
    CylFFTPreconditioner
    (
        std::shared_ptr<const gko::Executor> exec,
        gko::dim<2> size,
        int nr, int nSectors, int nthetaSector
    )
    :
        gko::EnableLinOp<CylFFTPreconditioner>(exec, size),
        totalCells_(nr * nSectors * nthetaSector),
        sectorMode_(true)
    {
        int useFloat = std::is_same<ValueType, float>::value ? 1 : 0;
        handle_ = cylFFTPrecondCreateSector(
            nr, nSectors, nthetaSector, useFloat);
    }


public:

    // Set tridiagonal coefficients from actual FV matrix couplings (DFT mode)
    void setCoeffs
    (
        const double* lower,
        const double* upper,
        const double* thetaCoeff
    )
    {
        if (handle_)
        {
            cylFFTPrecondSetCoeffs(handle_, lower, upper, thetaCoeff);
        }
    }

    // Set tridiagonal coefficients for per-sector DCT mode
    void setCoeffsSector
    (
        const double* lower,
        const double* upper,
        const double* thetaCoeff
    )
    {
        if (handle_)
        {
            cylFFTPrecondSetCoeffsSector(handle_, lower, upper, thetaCoeff);
        }
    }


private:

    CylFFTHandle handle_;
    int totalCells_;
    bool sectorMode_;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
