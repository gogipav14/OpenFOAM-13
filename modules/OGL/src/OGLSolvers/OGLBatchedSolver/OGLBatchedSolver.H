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
    Foam::OGL::OGLBatchedSolver

Description
    Batched GPU solver for solving multiple systems with the same matrix.

    Designed for velocity fields (U.x, U.y, U.z) which share the same
    coefficient matrix but have different RHS vectors. Solves all components
    in a single kernel launch for better GPU utilization.

    Benefits:
    - Single matrix setup for all components
    - Better GPU occupancy through batched operations
    - Reduced kernel launch overhead
    - Amortized halo exchange cost

    The solver uses Ginkgo's batched linear solver capabilities when
    available, with fallback to sequential solves if not.

    Usage:
    \verbatim
    solvers
    {
        "(U|k|epsilon)"
        {
            solver          OGLBatched;
            tolerance       1e-6;
            relTol          0.01;

            OGLCoeffs
            {
                batchSize       3;  // For vector fields
                precisionPolicy FP32;
            }
        }
    }
    \endverbatim

SourceFiles
    OGLBatchedSolver.C

\*---------------------------------------------------------------------------*/

#ifndef OGLBatchedSolver_H
#define OGLBatchedSolver_H

#include "OGLSolverBase.H"
#include "FoamGinkgoLinOp.H"

#include <ginkgo/ginkgo.hpp>
#include <vector>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                      Class OGLBatchedSolver Declaration
\*---------------------------------------------------------------------------*/

class OGLBatchedSolver
:
    public OGLSolverBase
{
    // Private Data

        //- Batch size (number of RHS to solve simultaneously)
        label batchSize_;

        //- Cached operator (shared for all batch components)
        mutable std::shared_ptr<FoamGinkgoLinOp<float>> operatorF32_;
        mutable std::shared_ptr<FoamGinkgoLinOp<double>> operatorF64_;

        //- Batched RHS storage
        mutable std::vector<scalarField> batchedRHS_;

        //- Batched solution storage
        mutable std::vector<scalarField> batchedSolution_;

        //- Current batch index
        mutable label currentBatchIndex_;

        //- Whether batched mode is active
        mutable bool batchedModeActive_;


    // Private Member Functions

        //- Initialize batch storage
        void initBatch(label nCells) const;

        //- Execute batched solve
        label executeBatchedSolve() const;

        //- Fallback to sequential solves
        label executeSequentialSolve() const;


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
    TypeName("OGLBatched");


    // Constructors

        //- Construct from matrix components and solver controls
        OGLBatchedSolver
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Disallow default bitwise copy construction
        OGLBatchedSolver(const OGLBatchedSolver&) = delete;


    //- Destructor
    virtual ~OGLBatchedSolver() = default;


    // Member Functions

        // Batch Control

            //- Start a new batch
            void beginBatch();

            //- Add a system to the current batch
            void addToBatch
            (
                scalarField& psi,
                const scalarField& source
            );

            //- Execute the batch and return total iterations
            label endBatch();

            //- Check if batched mode is active
            bool batchedModeActive() const { return batchedModeActive_; }


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const OGLBatchedSolver&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
