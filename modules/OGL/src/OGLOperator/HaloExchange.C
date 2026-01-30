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

\*---------------------------------------------------------------------------*/

#include "HaloExchange.H"
#include "lduInterfaceField.H"
#include "processorLduInterfaceField.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::OGL::HaloExchange::HaloExchange
(
    const lduMatrix& matrix,
    const lduInterfaceFieldPtrsList& interfaces
)
:
    matrix_(matrix),
    interfaces_(interfaces),
    pBufs_(Pstream::commsTypes::nonBlocking),
    sendBuffers_(interfaces.size()),
    recvBuffers_(interfaces.size()),
    inProgress_(false)
{
    // Pre-allocate buffers based on interface sizes
    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            const lduInterfaceField& intf = interfaces_[i];
            const label size = intf.interface().faceCells().size();
            sendBuffers_[i].setSize(size);
            recvBuffers_[i].setSize(size);
        }
    }
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::OGL::HaloExchange::initiate(const scalarField& x) const
{
    if (inProgress_)
    {
        FatalErrorInFunction
            << "Halo exchange already in progress"
            << abort(FatalError);
    }

    inProgress_ = true;

    // Clear previous buffers
    pBufs_.clear();

    // Pack and send for each interface
    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            const lduInterfaceField& intf = interfaces_[i];
            const labelUList& faceCells = intf.interface().faceCells();

            // Pack boundary values into send buffer
            scalarField& sendBuf = sendBuffers_[i];
            forAll(faceCells, fi)
            {
                sendBuf[fi] = x[faceCells[fi]];
            }

            // Initiate send via interface's mechanism
            // For processor interfaces, this starts MPI sends
            intf.initInterfaceMatrixUpdate
            (
                sendBuf,
                interfaces_,
                Pstream::defaultCommsType
            );
        }
    }

    // Finalize sends (for non-blocking communication)
    if (Pstream::parRun())
    {
        pBufs_.finishedSends();
    }
}


void Foam::OGL::HaloExchange::finalize() const
{
    if (!inProgress_)
    {
        return;
    }

    // Wait for receives and unpack
    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            const lduInterfaceField& intf = interfaces_[i];

            // Complete the interface update
            // For processor interfaces, this receives neighbor values
            intf.updateInterfaceMatrix
            (
                recvBuffers_[i],
                sendBuffers_[i],
                interfaces_,
                Pstream::defaultCommsType
            );
        }
    }

    inProgress_ = false;
}


// ************************************************************************* //
