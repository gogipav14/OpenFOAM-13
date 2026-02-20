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
    Foam::OGL::OGLBiCGStabSolver

Description
    GPU-accelerated BiCGStab solver for asymmetric lduMatrices using Ginkgo.

    Designed for momentum equations (U, k, epsilon) which produce asymmetric
    matrices due to advection terms. Uses gko::solver::Bicgstab with
    Block Jacobi or composite preconditioners.

    Features:
    - Mixed precision (FP32 solve with FP64 iterative refinement)
    - Correct interface handling via FoamGinkgoLinOp
    - Operator structure caching for performance
    - Multi-GPU ready via MPI rank-to-device mapping

    Configuration example:
    \verbatim
    solvers
    {
        U
        {
            solver          OGLBiCGStab;
            tolerance       1e-6;
            relTol          0.1;

            OGLCoeffs
            {
                precisionPolicy     FP64;
                preconditioner      blockJacobi;
                blockSize           4;
            }
        }
    }
    \endverbatim

SourceFiles
    OGLBiCGStabSolver.C

\*---------------------------------------------------------------------------*/

#ifndef OGLBiCGStabSolver_H
#define OGLBiCGStabSolver_H

#include "OGLSolverBase.H"
#include "FoamGinkgoLinOp.H"

#include "GinkgoCompat.H"
#include <map>
#include <mutex>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                    Class OGLBiCGStabSolver Declaration
\*---------------------------------------------------------------------------*/

class OGLBiCGStabSolver
:
    public OGLSolverBase
{
    // Private Types

        //- Cache entry for persisting GPU state across solver instantiations
        struct SolverCache
        {
            std::shared_ptr<FoamGinkgoLinOp<float>> opF32;
            std::shared_ptr<FoamGinkgoLinOp<double>> opF64;
        };


    // Private Static Data

        //- Static cache: field name -> GPU state (persists across
        //  OpenFOAM solver re-instantiations)
        static std::map<word, SolverCache> cache_;

        //- Mutex for cache access
        static std::mutex cacheMutex_;


    // Private Data

        //- Cached FP32 operator (references into static cache)
        mutable std::shared_ptr<FoamGinkgoLinOp<float>> operatorF32_;

        //- Cached FP64 operator
        mutable std::shared_ptr<FoamGinkgoLinOp<double>> operatorF64_;

        //- FP32 solver (rebuilt each call — momentum matrix changes frequently)
        mutable std::shared_ptr<gko::LinOp> solverF32_;

        //- FP64 solver
        mutable std::shared_ptr<gko::LinOp> solverF64_;


    // Private Member Functions

        //- Templated solver creation (reduces FP32/FP64 code duplication)
        template<typename ValueType>
        void updateSolverImpl
        (
            std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op,
            std::shared_ptr<gko::LinOp>& solver,
            ValueType tolerance
        ) const;

        //- Templated solve implementation with error handling
        template<typename ValueType>
        label solveImpl
        (
            scalarField& psi,
            const scalarField& source,
            std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op,
            std::shared_ptr<gko::LinOp>& solver,
            ValueType tolerance
        ) const;


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
    TypeName("OGLBiCGStab");


    // Constructors

        //- Construct from matrix components and solver controls
        OGLBiCGStabSolver
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Disallow default bitwise copy construction
        OGLBiCGStabSolver(const OGLBiCGStabSolver&) = delete;


    //- Destructor
    virtual ~OGLBiCGStabSolver() = default;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const OGLBiCGStabSolver&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
