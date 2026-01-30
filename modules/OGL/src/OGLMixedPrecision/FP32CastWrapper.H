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
    Foam::OGL::FP32CastWrapper

Description
    Utility class for casting between FP64 (OpenFOAM's scalar) and FP32
    for mixed-precision GPU solves.

    The standard HPC pattern for mixed-precision iterative solvers:
    1. Cast RHS and initial guess to FP32
    2. Solve in FP32 on GPU (with loose tolerance)
    3. Cast solution back to FP64
    4. Compute residual in FP64
    5. If not converged, iterate with correction

SourceFiles
    FP32CastWrapper.C

\*---------------------------------------------------------------------------*/

#ifndef FP32CastWrapper_H
#define FP32CastWrapper_H

#include "scalarField.H"

#include <ginkgo/ginkgo.hpp>
#include <memory>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                      Class FP32CastWrapper Declaration
\*---------------------------------------------------------------------------*/

class FP32CastWrapper
{
public:

    // Public Types

        using VectorF32 = gko::matrix::Dense<float>;
        using VectorF64 = gko::matrix::Dense<double>;


    // Static Member Functions

        //- Cast OpenFOAM scalarField to Ginkgo FP32 vector
        static std::shared_ptr<VectorF32> toGinkgoF32
        (
            std::shared_ptr<const gko::Executor> exec,
            const scalarField& field
        );

        //- Cast OpenFOAM scalarField to Ginkgo FP64 vector
        static std::shared_ptr<VectorF64> toGinkgoF64
        (
            std::shared_ptr<const gko::Executor> exec,
            const scalarField& field
        );

        //- Cast Ginkgo FP32 vector to OpenFOAM scalarField
        static void fromGinkgoF32
        (
            const VectorF32* gkoVec,
            scalarField& field
        );

        //- Cast Ginkgo FP64 vector to OpenFOAM scalarField
        static void fromGinkgoF64
        (
            const VectorF64* gkoVec,
            scalarField& field
        );

        //- Add FP32 correction to FP64 field: x += dx
        static void addCorrectionF32ToF64
        (
            const VectorF32* dxF32,
            scalarField& xF64
        );

        //- Cast FP64 vector to FP32 vector (on same executor)
        static std::shared_ptr<VectorF32> castF64ToF32
        (
            std::shared_ptr<const gko::Executor> exec,
            const VectorF64* vecF64
        );

        //- Cast FP32 vector to FP64 vector (on same executor)
        static std::shared_ptr<VectorF64> castF32ToF64
        (
            std::shared_ptr<const gko::Executor> exec,
            const VectorF32* vecF32
        );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
