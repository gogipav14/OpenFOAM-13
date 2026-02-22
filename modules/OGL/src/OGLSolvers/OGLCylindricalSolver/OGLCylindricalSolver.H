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
    Foam::OGL::OGLCylindricalSolver

Description
    GPU-accelerated cylindrical spectral pressure solver.

    PCG solver using DFT+Thomas preconditioner for cylindrical (r, theta)
    grids. The theta direction is periodic and decoupled by DFT. Each
    Fourier mode leaves a tridiagonal system in r with r-dependent
    coefficients, solved via pre-factored Thomas algorithm.

    The cylindrical mesh topology is auto-detected from cell centres:
    cells are transformed to (r, theta) coordinates relative to the
    specified axis. The structured (r, theta) grid is identified from
    unique coordinate counts, and a permutation map is built for
    gather/scatter between OpenFOAM and structured ordering.

    For the constant-coefficient case (uniform rAU), the preconditioner
    is exact. For variable coefficients, PCG iterates to correct.

    Configuration example:
    \verbatim
    solvers
    {
        p
        {
            solver          OGLCylindrical;
            tolerance       1e-6;
            relTol          0.01;

            OGLCoeffs
            {
                precisionPolicy     FP32;
                iterativeRefinement on;
                maxRefineIters      10;

                // Cylindrical axis (required)
                cylindricalOrigin   (0 0 0);
                cylindricalAxis     (0 0 1);

                // Optional: zone-based (only apply to named cellZone)
                // cylindricalZone   rotor;
            }
        }
    }
    \endverbatim

SourceFiles
    OGLCylindricalSolver.C

\*---------------------------------------------------------------------------*/

#ifndef OGLCylindricalSolver_H
#define OGLCylindricalSolver_H

#include "OGLSolverBase.H"
#include "CylFFTPreconditioner.H"

#include "GinkgoCompat.H"
#include <map>
#include <mutex>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                    Class OGLCylindricalSolver Declaration
\*---------------------------------------------------------------------------*/

class OGLCylindricalSolver
:
    public OGLSolverBase
{
    // Private Types

        //- Cache entry for persisting GPU state across solver instantiations
        struct CylCache
        {
            std::shared_ptr<CylFFTPreconditioner<float>> cylF32;
            std::shared_ptr<CylFFTPreconditioner<double>> cylF64;
            bool coeffsInitialized;
        };


    // Private Static Data

        static std::map<word, CylCache> cache_;
        static std::mutex cacheMutex_;


    // Private Data

        //- Cylindrical axis origin
        vector cylindricalOrigin_;

        //- Cylindrical axis direction (unit vector)
        vector cylindricalAxis_;

        //- Optional: zone name for zone-based preconditioning
        word cylindricalZoneName_;

        //- Whether zone mode is active
        bool useZone_;

        //- Detected cylindrical grid dimensions
        mutable label nr_;       // radial cells
        mutable label ntheta_;   // angular cells

        //- Permutation: OpenFOAM cell index -> structured (r, theta) index
        //  permutation_[i] = structured index for OpenFOAM cell i
        //  structuredToFoam_[j] = OpenFOAM cell index for structured index j
        mutable labelList permutation_;
        mutable labelList structuredToFoam_;

        //- Zone cell indices (sorted)
        mutable labelList zoneCells_;

        //- Whether cylindrical detection has been done
        mutable bool detected_;

        //- Cached preconditioner
        mutable std::shared_ptr<CylFFTPreconditioner<float>> cylF32_;
        mutable std::shared_ptr<CylFFTPreconditioner<double>> cylF64_;

        //- Whether coefficients have been initialized
        mutable bool coeffsInitialized_;


    // Private Member Functions

        //- Detect cylindrical mesh structure from cell centres.
        //  Computes (r, theta) for each cell, identifies nr and ntheta
        //  from unique coordinate counts, and builds the permutation map.
        bool detectCylindricalMesh() const;

        //- Extract radial/angular coupling coefficients from the ldu matrix
        //  and update the cylindrical preconditioner's Thomas factors.
        void extractCylindricalCoeffs() const;

        //- Apply cylindrical preconditioner: gather -> DFT+Thomas -> scatter
        template<typename ValueType>
        void applyCylPrecond
        (
            scalarField& z,
            const scalarField& r,
            std::shared_ptr<CylFFTPreconditioner<ValueType>>& precond
        ) const;


protected:

    // Protected Member Functions

        virtual label solveFP32
        (
            scalarField& psi,
            const scalarField& source,
            const scalar tolerance
        ) const override;

        virtual label solveFP64
        (
            scalarField& psi,
            const scalarField& source,
            const scalar tolerance
        ) const override;


public:

    //- Runtime type information
    TypeName("OGLCylindrical");


    // Constructors

        OGLCylindricalSolver
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        OGLCylindricalSolver(const OGLCylindricalSolver&) = delete;


    //- Destructor
    virtual ~OGLCylindricalSolver() = default;


    // Member Functions

        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt = 0
        ) const override;

        void operator=(const OGLCylindricalSolver&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
