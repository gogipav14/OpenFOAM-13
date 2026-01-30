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
    Foam::OGL::HaloExchange

Description
    Manages distributed vector halo exchange for multi-GPU execution.
    Provides abstraction for:
    - LocalVector: owned cells on this rank
    - HaloVector: ghost cells needed from neighbors

    This class handles packing boundary values, MPI communication,
    and unpacking received values for interface computations.

SourceFiles
    HaloExchange.C

\*---------------------------------------------------------------------------*/

#ifndef HaloExchange_H
#define HaloExchange_H

#include "lduMatrix.H"
#include "lduInterfaceFieldPtrsList.H"
#include "PstreamBuffers.H"
#include "scalarField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                       Class HaloExchange Declaration
\*---------------------------------------------------------------------------*/

class HaloExchange
{
    // Private Data

        //- Reference to lduMatrix for mesh information
        const lduMatrix& matrix_;

        //- Interface field pointers
        const lduInterfaceFieldPtrsList& interfaces_;

        //- Pstream buffers for non-blocking communication
        mutable PstreamBuffers pBufs_;

        //- Send buffers for each interface
        mutable List<scalarField> sendBuffers_;

        //- Receive buffers for each interface
        mutable List<scalarField> recvBuffers_;

        //- Whether exchange is currently in progress
        mutable bool inProgress_;


    // Private Member Functions

        //- Disallow default bitwise copy construction
        HaloExchange(const HaloExchange&) = delete;

        //- Disallow default bitwise assignment
        void operator=(const HaloExchange&) = delete;


public:

    // Constructors

        //- Construct from matrix and interfaces
        HaloExchange
        (
            const lduMatrix& matrix,
            const lduInterfaceFieldPtrsList& interfaces
        );


    //- Destructor
    ~HaloExchange() = default;


    // Member Functions

        //- Initiate non-blocking halo exchange
        //  Packs boundary values from x and starts MPI sends
        void initiate(const scalarField& x) const;

        //- Finalize halo exchange
        //  Waits for MPI receives and makes halo values available
        void finalize() const;

        //- Check if exchange is in progress
        bool inProgress() const { return inProgress_; }

        //- Access received halo values for an interface
        const scalarField& haloValues(const label interfacei) const
        {
            return recvBuffers_[interfacei];
        }

        //- Number of interfaces
        label nInterfaces() const
        {
            return interfaces_.size();
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
