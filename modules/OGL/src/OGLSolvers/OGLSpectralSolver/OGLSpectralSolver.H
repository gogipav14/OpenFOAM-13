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
    Foam::OGL::OGLSpectralSolver

Description
    GPU-accelerated spectral pressure solver with optional zone-based
    additive Schwarz domain decomposition.

    PCG solver using DCT-II preconditioner via cuFFT. For the constant-
    coefficient Laplacian on a uniform Cartesian grid with Neumann BCs,
    the DCT diagonalization is exact. For variable coefficients (rAU from
    momentum), PCG iterates to correct the discrepancy.

    Zone decomposition (Phase 3):
    When spectralZone is specified, the DCT is applied only to cells in
    the named cellZone (the "structured" region). Non-zone cells receive
    diagonal (Jacobi) preconditioning. This is an additive Schwarz
    preconditioner where PCG handles the zone boundary coupling.

    Configuration example:
    \verbatim
    solvers
    {
        p
        {
            solver          OGLSpectral;
            tolerance       1e-6;
            relTol          0.01;

            OGLCoeffs
            {
                precisionPolicy     FP32;
                iterativeRefinement on;
                maxRefineIters      10;
                fftDimensions       (200 200 1);
                meshSpacing         (0.005 0.005 0.01);

                // Optional: zone-based domain decomposition
                // spectralZone    innerRegion;
            }
        }
    }
    \endverbatim

SourceFiles
    OGLSpectralSolver.C

\*---------------------------------------------------------------------------*/

#ifndef OGLSpectralSolver_H
#define OGLSpectralSolver_H

#include "OGLSolverBase.H"
#include "FoamGinkgoLinOp.H"
#include "FFTPreconditioner.H"

#include "GinkgoCompat.H"
#include <map>
#include <mutex>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                     Class OGLSpectralSolver Declaration
\*---------------------------------------------------------------------------*/

class OGLSpectralSolver
:
    public OGLSolverBase
{
    // Private Types

        //- Cache entry for persisting GPU state across solver instantiations
        struct SpectralCache
        {
            std::shared_ptr<FoamGinkgoLinOp<float>> opF32;
            std::shared_ptr<FoamGinkgoLinOp<double>> opF64;
            std::shared_ptr<FFTPreconditioner<float>> fftF32;
            std::shared_ptr<FFTPreconditioner<double>> fftF64;
            bool coeffsInitialized;
        };


    // Private Static Data

        //- Static cache: field name -> GPU state
        static std::map<word, SpectralCache> cache_;

        //- Mutex for cache access
        static std::mutex cacheMutex_;


    // Private Data

        //- Cached FP32 operator (for CSR coefficient extraction)
        mutable std::shared_ptr<FoamGinkgoLinOp<float>> operatorF32_;

        //- Cached FP64 operator
        mutable std::shared_ptr<FoamGinkgoLinOp<double>> operatorF64_;

        //- Cached FP32 FFT direct solver
        mutable std::shared_ptr<FFTPreconditioner<float>> fftSolverF32_;

        //- Cached FP64 FFT direct solver
        mutable std::shared_ptr<FFTPreconditioner<double>> fftSolverF64_;

        //- Whether coefficients have been initialized from the CSR matrix
        mutable bool coeffsInitialized_;


    // Zone decomposition data (additive Schwarz)

        //- Name of the cellZone for spectral preconditioning (empty = off)
        word spectralZoneName_;

        //- Whether zone decomposition is active
        bool useZone_;

        //- Sorted zone cell indices (DCT ordering = sorted blockMesh order)
        mutable labelList zoneCellsSorted_;

        //- Non-zone cell indices (receive Jacobi preconditioning)
        mutable labelList nonZoneCells_;

        //- Whether zone mappings have been initialized
        mutable bool zoneInitialized_;


    // Private Member Functions

        //- Initialize zone cell mappings from the mesh cellZone
        void initZone() const;

        //- Apply zone-based additive Schwarz preconditioner:
        //  zone cells get DCT, non-zone cells get Jacobi
        void applyZonePreconditioner
        (
            scalarField& z,
            const scalarField& r
        ) const;

        //- Ensure the FoamGinkgoLinOp exists and has current values
        template<typename ValueType>
        void ensureOperator
        (
            std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op
        ) const;

        //- Create FFT solver and extract coupling coefficients from CSR
        template<typename ValueType>
        void initFFTSolver
        (
            std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op,
            std::shared_ptr<FFTPreconditioner<ValueType>>& fftSolver
        ) const;

        //- Apply FFT direct solve: x = L^{-1} * b (full setup + solve)
        template<typename ValueType>
        label spectralSolveImpl
        (
            scalarField& psi,
            const scalarField& source,
            std::shared_ptr<FoamGinkgoLinOp<ValueType>>& op,
            std::shared_ptr<FFTPreconditioner<ValueType>>& fftSolver
        ) const;

        //- Apply DCT solve only (no operator/coefficient setup).
        //  Used inside the defect correction loop where setup is done once.
        template<typename ValueType>
        void applyDCT
        (
            scalarField& psi,
            const scalarField& source,
            std::shared_ptr<FFTPreconditioner<ValueType>>& fftSolver
        ) const;


protected:

    // Protected Member Functions

        //- Perform single precision solve (DCT direct)
        virtual label solveFP32
        (
            scalarField& psi,
            const scalarField& source,
            const scalar tolerance
        ) const override;

        //- Perform double precision solve (DCT direct)
        virtual label solveFP64
        (
            scalarField& psi,
            const scalarField& source,
            const scalar tolerance
        ) const override;


public:

    //- Runtime type information
    TypeName("OGLSpectral");


    // Constructors

        //- Construct from matrix components and solver controls
        OGLSpectralSolver
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Disallow default bitwise copy construction
        OGLSpectralSolver(const OGLSpectralSolver&) = delete;


    //- Destructor
    virtual ~OGLSpectralSolver() = default;


    // Member Functions

        //- Solve with PCG using DCT preconditioner
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt = 0
        ) const override;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const OGLSpectralSolver&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
