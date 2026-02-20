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
    Foam::OGL::AdditiveLinOp

Description
    Ginkgo LinOp that applies two preconditioners additively:
        apply(b, x) computes x = op1(b) + op2(b)

    This preserves SPD when both op1 and op2 are SPD, since the sum of
    two SPD operators is SPD.

\*---------------------------------------------------------------------------*/

#ifndef AdditiveLinOp_H
#define AdditiveLinOp_H

#include "GinkgoCompat.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

template <typename ValueType>
class AdditiveLinOp
:
    public gko::EnableLinOp<AdditiveLinOp<ValueType>>,
    public gko::EnableCreateMethod<AdditiveLinOp<ValueType>>
{
    friend class gko::EnablePolymorphicObject<AdditiveLinOp, gko::LinOp>;
    friend class gko::EnableCreateMethod<AdditiveLinOp>;

public:

    using value_type = ValueType;


protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        // x = op1(b)
        op1_->apply(b, x);

        // temp = op2(b)
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);
        auto temp = gko::matrix::Dense<ValueType>::create(
            this->get_executor(), dense_x->get_size()
        );
        op2_->apply(b, temp.get());

        // x = x + temp
        auto one = gko::initialize<gko::matrix::Dense<ValueType>>(
            {ValueType{1.0}}, this->get_executor()
        );
        dense_x->add_scaled(one, temp);
    }

    void apply_impl
    (
        const gko::LinOp* alpha,
        const gko::LinOp* b,
        const gko::LinOp* beta,
        gko::LinOp* x
    ) const override
    {
        auto dense_x = gko::as<gko::matrix::Dense<ValueType>>(x);

        // temp = op1(b) + op2(b)
        auto temp = gko::matrix::Dense<ValueType>::create(
            this->get_executor(), dense_x->get_size()
        );
        op1_->apply(b, temp.get());

        auto temp2 = gko::matrix::Dense<ValueType>::create(
            this->get_executor(), dense_x->get_size()
        );
        op2_->apply(b, temp2.get());

        auto one = gko::initialize<gko::matrix::Dense<ValueType>>(
            {ValueType{1.0}}, this->get_executor()
        );
        temp->add_scaled(one, temp2);

        // x = alpha * temp + beta * x
        auto dense_alpha =
            gko::as<gko::matrix::Dense<ValueType>>(alpha);
        auto dense_beta =
            gko::as<gko::matrix::Dense<ValueType>>(beta);
        dense_x->scale(dense_beta);
        dense_x->add_scaled(dense_alpha, temp);
    }

    // Default constructor (required by EnableCreateMethod)
    explicit AdditiveLinOp(std::shared_ptr<const gko::Executor> exec)
    :
        gko::EnableLinOp<AdditiveLinOp>(exec)
    {}

    // Construct from two operators
    AdditiveLinOp
    (
        std::shared_ptr<const gko::LinOp> op1,
        std::shared_ptr<const gko::LinOp> op2
    )
    :
        gko::EnableLinOp<AdditiveLinOp>(
            op1->get_executor(), op1->get_size()
        ),
        op1_(std::move(op1)),
        op2_(std::move(op2))
    {}


private:

    std::shared_ptr<const gko::LinOp> op1_;
    std::shared_ptr<const gko::LinOp> op2_;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
