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
#include "DICPreconditioner.H"

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

        //- Fallback preconditioner for non-zone cells
        enum FallbackType
        {
            JACOBI,
            DIC
        };

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

        //- Fallback preconditioner type for non-zone cells
        FallbackType fallbackType_;

        //- DIC reciprocal diagonal (recomputed each solve call)
        mutable scalarField rDDIC_;

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

        // --- Capacitance (Woodbury) correction data ---

        //- Number of defect angular faces (rank of correction)
        mutable label capK_;

        //- Structured indices of defect face cell pairs
        //  capCellA_[f], capCellB_[f]: structured indices for defect face f
        mutable labelList capCellA_;
        mutable labelList capCellB_;

        //- Coupling coefficient that M assumes but A lacks (per defect face)
        mutable scalarList capCoeffs_;

        //- W = M^{-1} U precomputed columns (capK_ columns of length nStruct)
        mutable scalarList capW_;

        //- Factored capacitance matrix S^{-1} (capK_ x capK_ dense)
        mutable scalarList capSinv_;

        //- Whether capacitance correction is active
        mutable bool capValid_;

        // --- Per-sector DCT mode ---

        //- Whether to use per-sector DCT instead of global DFT+Woodbury
        bool useSectorDCT_;

        //- Number of angular sectors (= number of blade walls)
        mutable label nSectors_;

        //- Angular cells per sector
        mutable label nthetaSector_;

        //- Sector start theta indices (theta index of first cell in each sector)
        //  sectorStarts_[s] = first theta index of sector s
        mutable labelList sectorStarts_;

        //- Permutation: structured index -> sector-ordered index
        //  Sector-ordered: data[(s*nr + ir)*nthetaSector + itheta_local]
        mutable labelList structToSector_;

        //- Inverse permutation: sector-ordered -> structured index
        mutable labelList sectorToStruct_;

        //- Whether sector detection has been done
        mutable bool sectorsDetected_;

    // Private Member Functions

        //- Detect cylindrical mesh structure from cell centres.
        //  Computes (r, theta) for each cell, identifies nr and ntheta
        //  from unique coordinate counts, and builds the permutation map.
        bool detectCylindricalMesh() const;

        //- Extract radial/angular coupling coefficients from the ldu matrix
        //  and update the cylindrical preconditioner's Thomas factors.
        void extractCylindricalCoeffs() const;

        //- Build capacitance matrix for Woodbury blade-wall correction.
        //  Identifies angular faces where the actual coupling differs from
        //  the mean, precomputes W = M^{-1}*U, and factors the k×k
        //  capacitance matrix S = C_diag^{-1} + U^T*W.
        void buildCapacitanceMatrix() const;

        //- Apply spectral solve to a single column (CPU-side helper)
        void applySpectralColumn
        (
            scalarField& z,
            const scalarField& r
        ) const;

        //- Apply cylindrical preconditioner: gather -> DFT+Thomas -> scatter
        template<typename ValueType>
        void applyCylPrecond
        (
            scalarField& z,
            const scalarField& r,
            std::shared_ptr<CylFFTPreconditioner<ValueType>>& precond
        ) const;

        //- Detect sector boundaries from blade-wall face detection.
        //  Uses the pairExists data from extractCylindricalCoeffs().
        //  Populates nSectors_, nthetaSector_, sectorStarts_, and
        //  structToSector_/sectorToStruct_ permutations.
        void detectSectors() const;

        //- Apply per-sector DCT preconditioner: gather to sector-ordered
        //  layout -> DCT+Thomas per sector -> scatter back.
        //  No Woodbury correction needed.
        template<typename ValueType>
        void applyCylPrecondSector
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
