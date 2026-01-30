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
    Foam::OGL::OGLSolverBase

Description
    Base class for GPU-accelerated linear solvers using Ginkgo.

    Provides:
    - Mixed precision support (FP32 solve + FP64 residual)
    - Iterative refinement for mathematical correctness
    - Integration with OpenFOAM's lduMatrix solver interface
    - Structure/value caching options for performance

    Derived classes implement specific solver algorithms (CG, BiCGStab, etc.)

SourceFiles
    OGLSolverBase.C

\*---------------------------------------------------------------------------*/

#ifndef OGLSolverBase_H
#define OGLSolverBase_H

#include "lduMatrix.H"
#include "FoamGinkgoLinOp.H"
#include "FP32CastWrapper.H"
#include "PerformanceTimer.H"

#include <ginkgo/ginkgo.hpp>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                       Class OGLSolverBase Declaration
\*---------------------------------------------------------------------------*/

class OGLSolverBase
:
    public lduMatrix::solver
{
public:

    // Public Enumerations

        //- Precision policy for GPU solve
        enum class PrecisionPolicy
        {
            FP64,       // Full double precision
            FP32,       // Single precision with iterative refinement
            MIXED       // FP32 solve, FP64 residual check
        };


protected:

    // Protected Data

        //- Precision policy
        PrecisionPolicy precisionPolicy_;

        //- Enable iterative refinement
        bool iterativeRefinement_;

        //- Maximum refinement iterations
        label maxRefineIters_;

        //- Inner tolerance for FP32 solve (looser than final tolerance)
        scalar innerTolerance_;

        //- Cache structure between solves
        bool cacheStructure_;

        //- Cache values between solves
        bool cacheValues_;

        //- Debug level
        label debug_;

        //- Enable timing instrumentation
        bool timing_;

        //- Performance timer
        mutable PerformanceTimer timer_;


    // Protected Member Functions

        //- Read solver controls from dictionary
        void readOGLControls();

        //- Compute residual norm: ||b - A*x|| / normFactor
        scalar computeResidualNorm
        (
            const scalarField& psi,
            const scalarField& source,
            const direction cmpt
        ) const;

        //- Solve with iterative refinement (FP32 solve + FP64 correction)
        solverPerformance solveWithRefinement
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt
        ) const;

        //- Pure virtual: perform single precision solve
        //  Returns iteration count for this refinement step
        virtual label solveFP32
        (
            scalarField& psi,
            const scalarField& residual,
            const scalar tolerance
        ) const = 0;

        //- Pure virtual: perform double precision solve
        virtual label solveFP64
        (
            scalarField& psi,
            const scalarField& source,
            const scalar tolerance
        ) const = 0;


public:

    // Constructors

        //- Construct from matrix components and solver controls
        OGLSolverBase
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Disallow default bitwise copy construction
        OGLSolverBase(const OGLSolverBase&) = delete;


    //- Destructor
    virtual ~OGLSolverBase() = default;


    // Member Functions

        //- Solve the matrix with this solver
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt = 0
        ) const override;

        //- Access performance timer
        PerformanceTimer& timer() const { return timer_; }

        //- Print performance report
        void reportPerformance() const;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const OGLSolverBase&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
