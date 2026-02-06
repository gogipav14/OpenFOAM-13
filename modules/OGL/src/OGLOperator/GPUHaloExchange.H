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
    Foam::OGL::GPUHaloExchange

Description
    GPU-resident halo exchange for multi-GPU execution.

    Eliminates the GPU<->CPU transfer bottleneck by:
    1. Keeping vectors resident on GPU throughout the solve
    2. Only transferring boundary values (not entire vectors)
    3. Using CUDA-aware MPI when available for direct GPU-to-GPU transfer
    4. Falling back to staged host transfers when CUDA-aware MPI unavailable

    The exchange workflow is:
    1. gather(): Extract boundary cell values from GPU vector into send buffer
    2. exchange(): MPI send/receive of boundary values
    3. scatter(): Add received halo contributions to GPU result vector

    Performance compared to CPU-based approach:
    - CPU: 3 full vector transfers per apply (O(N) data movement)
    - GPU: 2 boundary transfers per apply (O(boundary) data movement)

SourceFiles
    GPUHaloExchange.C

\*---------------------------------------------------------------------------*/

#ifndef GPUHaloExchange_H
#define GPUHaloExchange_H

#include "lduMatrix.H"
#include "lduInterfaceFieldPtrsList.H"
#include "FieldField.H"
#include "scalarField.H"

#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <vector>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                     Class GPUHaloExchange Declaration
\*---------------------------------------------------------------------------*/

template<typename ValueType>
class GPUHaloExchange
{
public:

    // Public Types

        using Vector = gko::matrix::Dense<ValueType>;


private:

    // Private Data

        //- Ginkgo executor (GPU)
        std::shared_ptr<const gko::Executor> exec_;

        //- CPU executor for staging
        std::shared_ptr<const gko::Executor> cpuExec_;

        //- Reference to lduMatrix for mesh information
        const lduMatrix& matrix_;

        //- Interface field pointers
        const lduInterfaceFieldPtrsList& interfaces_;

        //- Interface boundary coefficients
        const FieldField<Field, scalar>& interfaceBouCoeffs_;

        //- Direction component
        direction cmpt_;

        //- Face cell indices for each interface (GPU-resident)
        std::vector<gko::array<int>> faceCellsGPU_;

        //- Send buffers on GPU (one per interface)
        mutable std::vector<std::shared_ptr<Vector>> sendBuffersGPU_;

        //- Receive buffers on GPU (one per interface)
        mutable std::vector<std::shared_ptr<Vector>> recvBuffersGPU_;

        //- Host staging buffers for non-CUDA-aware MPI
        mutable std::vector<scalarField> sendBuffersHost_;
        mutable std::vector<scalarField> recvBuffersHost_;

        //- Boundary coefficients on GPU (one per interface)
        std::vector<std::shared_ptr<Vector>> bouCoeffsGPU_;

        //- Whether CUDA-aware MPI is available
        bool cudaAwareMPI_;

        //- Total number of boundary faces
        label totalBoundaryFaces_;


    // Private Member Functions

        //- Initialize GPU buffers and face cell mappings
        void initializeBuffers();

        //- Detect if CUDA-aware MPI is available
        static bool detectCudaAwareMPI();

        //- Disallow default bitwise copy construction
        GPUHaloExchange(const GPUHaloExchange&) = delete;

        //- Disallow default bitwise assignment
        void operator=(const GPUHaloExchange&) = delete;


public:

    // Constructors

        //- Construct from matrix, interfaces, and executor
        GPUHaloExchange
        (
            std::shared_ptr<const gko::Executor> exec,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const direction cmpt = 0
        );


    //- Destructor
    ~GPUHaloExchange() = default;


    // Member Functions

        // Access

            //- Return total number of boundary faces
            label totalBoundaryFaces() const { return totalBoundaryFaces_; }

            //- Check if using CUDA-aware MPI
            bool cudaAwareMPI() const { return cudaAwareMPI_; }


        // Operations

            //- Gather boundary values from GPU vector into send buffers
            //  Extracts x[faceCells[i]] for each interface
            void gather(const Vector* x) const;

            //- Exchange halo data via MPI
            //  Uses CUDA-aware MPI if available, else stages through host
            void exchange() const;

            //- Scatter received halo contributions into result vector
            //  Computes y[faceCells[i]] += bouCoeffs[i] * recvBuffer[i]
            void scatter(Vector* y) const;

            //- Combined gather + exchange + scatter operation
            //  Convenience method for typical use case
            void apply(const Vector* x, Vector* y) const;


        // Diagnostics

            //- Report halo exchange statistics
            void report() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
