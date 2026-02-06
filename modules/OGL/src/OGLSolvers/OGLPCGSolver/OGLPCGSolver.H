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
    Foam::OGL::OGLPCGSolver

Description
    GPU-accelerated Preconditioned Conjugate Gradient solver for symmetric
    lduMatrices using Ginkgo.

    Features:
    - Mixed precision (FP32 solve with FP64 iterative refinement)
    - Correct interface handling via FoamGinkgoLinOp
    - Structure/value caching for performance
    - Multi-GPU ready via MPI rank-to-device mapping

    Configuration example:
    \verbatim
    solvers
    {
        p
        {
            solver          OGLPCG;
            tolerance       1e-6;
            relTol          0.01;

            OGLCoeffs
            {
                precisionPolicy     FP32;
                iterativeRefinement on;
                maxRefineIters      3;
                innerTolerance      1e-4;
                cacheStructure      true;
                cacheValues         false;
            }
        }
    }
    \endverbatim

SourceFiles
    OGLPCGSolver.C

\*---------------------------------------------------------------------------*/

#ifndef OGLPCGSolver_H
#define OGLPCGSolver_H

#include "OGLSolverBase.H"
#include "FoamGinkgoLinOp.H"

#include <ginkgo/ginkgo.hpp>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                        Class OGLPCGSolver Declaration
\*---------------------------------------------------------------------------*/

class OGLPCGSolver
:
    public OGLSolverBase
{
    // Private Data

        //- Cached FP32 operator
        mutable std::shared_ptr<FoamGinkgoLinOp<float>> operatorF32_;

        //- Cached FP64 operator
        mutable std::shared_ptr<FoamGinkgoLinOp<double>> operatorF64_;

        //- Cached FP32 solver
        mutable std::shared_ptr<gko::solver::Cg<float>> solverF32_;

        //- Cached FP64 solver
        mutable std::shared_ptr<gko::solver::Cg<double>> solverF64_;


    // Private Member Functions

        //- Templated solver creation (reduces FP32/FP64 code duplication)
        template<typename ValueType>
        void updateSolverImpl
        (
            std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op,
            std::shared_ptr<gko::solver::Cg<ValueType>>& solver,
            ValueType tolerance
        ) const;

        //- Templated solve implementation with error handling
        template<typename ValueType>
        label solveImpl
        (
            scalarField& psi,
            const scalarField& source,
            std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op,
            std::shared_ptr<gko::solver::Cg<ValueType>>& solver,
            ValueType tolerance
        ) const;

        //- Create/update FP32 operator and solver
        void updateSolverF32() const;

        //- Create/update FP64 operator and solver
        void updateSolverF64() const;


protected:

    // Protected Member Functions

        //- Perform single precision solve
        virtual label solveFP32
        (
            scalarField& psi,
            const scalarField& source,
            const scalar tolerance
        ) const override;

        //- Perform double precision solve
        virtual label solveFP64
        (
            scalarField& psi,
            const scalarField& source,
            const scalar tolerance
        ) const override;


public:

    //- Runtime type information
    TypeName("OGLPCG");


    // Constructors

        //- Construct from matrix components and solver controls
        OGLPCGSolver
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Disallow default bitwise copy construction
        OGLPCGSolver(const OGLPCGSolver&) = delete;


    //- Destructor
    virtual ~OGLPCGSolver() = default;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const OGLPCGSolver&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
